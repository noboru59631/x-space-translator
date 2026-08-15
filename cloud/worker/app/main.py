"""Cloud Run transcription proof-of-concept worker."""

from __future__ import annotations

import hmac
import importlib.util
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import psutil
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.services.audio import convert_to_wav, find_ffmpeg, validate_upload
from app.services.downloader import download_space, normalize_x_url
from app.services.errors import AppError, DownloadError
from app.services.transcriber import Transcriber

WORK_ROOT = Path(os.getenv("WORKER_TEMP_DIR", "/tmp/x-space-worker"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = min(25, max(1, int(os.getenv("MAX_UPLOAD_MB", "25"))))
RESULT_TTL_SECONDS = max(60, int(os.getenv("RESULT_TTL_SECONDS", "3600")))
MAX_JSON_BODY_BYTES = 8192
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
AVAILABLE_CPUS = max(1, os.cpu_count() or 1)
WHISPER_CPU_THREADS = min(
    AVAILABLE_CPUS,
    max(1, int(os.getenv("WHISPER_CPU_THREADS", str(AVAILABLE_CPUS)))),
)
LOGGER = logging.getLogger(__name__)


class BusyError(RuntimeError):
    """Raised when the single PoC worker already has an active job."""


class TranscribeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        try:
            return normalize_x_url(value)
        except AppError as exc:
            raise ValueError(str(exc)) from exc


class JobManager:
    """Run one CPU-heavy transcription at a time and keep results in memory."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()
        self.active_job_id: str | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cloud-worker")
        self.transcriber = Transcriber()

    def is_busy(self) -> bool:
        with self.lock:
            return self.active_job_id is not None

    def submit_url(self, url: str) -> dict[str, object]:
        job = self._reserve_job("url", url)
        self.executor.submit(self._run_url, str(job["job_id"]), url)
        return self.status_view(job)

    def reserve_file(self, filename: str) -> dict[str, object]:
        return self._reserve_job("file", filename)

    def dispatch_file(self, job: dict[str, object], source: Path) -> dict[str, object]:
        self.executor.submit(self._run_file, str(job["job_id"]), source)
        return self.status_view(job)

    def abandon(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id, None)
            if self.active_job_id == job_id:
                self.active_job_id = None

    def get(self, job_id: str) -> dict[str, object] | None:
        self._expire_old_jobs()
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    @staticmethod
    def status_view(job: dict[str, object]) -> dict[str, object]:
        return {
            key: job.get(key)
            for key in (
                "job_id",
                "status",
                "stage",
                "progress",
                "error_code",
                "error",
                "elapsed_seconds",
                "peak_memory_mb",
                "audio_deleted",
            )
        }

    @classmethod
    def result_view(cls, job: dict[str, object]) -> dict[str, object]:
        if job.get("status") == "completed" and isinstance(job.get("result"), dict):
            return dict(job["result"])
        return cls.status_view(job)

    def _reserve_job(self, source_type: str, source: str) -> dict[str, object]:
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "source_type": source_type,
            "source": source,
            "created_at": time.time(),
            "error_code": None,
            "error": None,
            "elapsed_seconds": None,
            "peak_memory_mb": None,
            "audio_deleted": False,
            "result": None,
        }
        with self.lock:
            if self.active_job_id is not None:
                raise BusyError("Worker is already processing another job")
            self.active_job_id = job_id
            self.jobs[job_id] = job
        return job

    def _update(self, job_id: str, **values: object) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(values)

    def _finish(self, job_id: str) -> None:
        with self.lock:
            if self.active_job_id == job_id:
                self.active_job_id = None

    def _run_url(self, job_id: str, url: str) -> None:
        job_dir = WORK_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        try:
            self._update(job_id, status="processing", stage="downloading", progress=5)
            source, metadata = download_space(url, job_dir)
            self._transcribe(
                job_id,
                source,
                job_dir,
                title=str(metadata.get("title") or ""),
                source_url=url,
            )
        except DownloadError as exc:
            self._fail(job_id, classify_download_failure(exc), str(exc))
        except AppError as exc:
            self._fail(job_id, "CODE_ERROR", str(exc))
        except Exception:
            LOGGER.exception("Unexpected URL job failure: %s", job_id)
            self._fail(job_id, "CODE_ERROR", "Unexpected worker error")
        finally:
            stop_sampling.set()
            shutil.rmtree(job_dir, ignore_errors=True)
            self._update_metrics(job_id, job_dir, started, peak)
            self._finish(job_id)

    def _run_file(self, job_id: str, source: Path) -> None:
        job_dir = source.parent
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        try:
            self._transcribe(
                job_id,
                source,
                job_dir,
                title=source.name,
                source_url="local-upload",
            )
        except AppError as exc:
            self._fail(job_id, "CODE_ERROR", str(exc))
        except Exception:
            LOGGER.exception("Unexpected upload job failure: %s", job_id)
            self._fail(job_id, "CODE_ERROR", "Unexpected worker error")
        finally:
            stop_sampling.set()
            shutil.rmtree(job_dir, ignore_errors=True)
            self._update_metrics(job_id, job_dir, started, peak)
            self._finish(job_id)

    def _transcribe(
        self,
        job_id: str,
        source: Path,
        job_dir: Path,
        *,
        title: str,
        source_url: str,
    ) -> None:
        self._update(job_id, status="processing", stage="converting", progress=20)
        wav_path = convert_to_wav(source, job_dir / "audio.wav")
        self._update(job_id, stage="transcribing", progress=35)
        raw = self.transcriber.transcribe(
            wav_path,
            mode="cloud",
            device_preference="cpu",
            model_override=WHISPER_MODEL,
            compute_override=WHISPER_COMPUTE_TYPE,
            progress=lambda value: self._update(job_id, progress=value),
            cpu_threads=WHISPER_CPU_THREADS,
        )
        result = {
            "status": "completed",
            "title": title,
            "source_url": source_url,
            "detected_language": raw.get("detected_language") or "",
            "duration": round(float(raw.get("duration") or 0), 3),
            "segments": [
                {
                    "start": round(float(segment["start"]), 3),
                    "end": round(float(segment["end"]), 3),
                    "speaker": "Speaker",
                    "original": str(segment["original"]),
                }
                for segment in raw.get("segments", [])
            ],
        }
        self._update(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            result=result,
        )

    def _update_metrics(
        self,
        job_id: str,
        job_dir: Path,
        started: float,
        peak: list[float],
    ) -> None:
        self._update(
            job_id,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            peak_memory_mb=round(peak[0], 1),
            audio_deleted=not job_dir.exists(),
        )

    def _fail(self, job_id: str, code: str, message: str) -> None:
        self._update(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            error_code=code,
            error=message,
        )

    @staticmethod
    def _start_memory_sampler() -> tuple[threading.Event, list[float]]:
        stopped = threading.Event()
        peak = [psutil.Process().memory_info().rss / 1024 / 1024]
        process = psutil.Process()

        def sample() -> None:
            while not stopped.wait(0.25):
                try:
                    rss = process.memory_info().rss
                    rss += sum(
                        child.memory_info().rss for child in process.children(recursive=True)
                    )
                    peak[0] = max(peak[0], rss / 1024 / 1024)
                except (psutil.Error, OSError):
                    pass

        threading.Thread(target=sample, daemon=True, name="memory-sampler").start()
        return stopped, peak

    def _expire_old_jobs(self) -> None:
        cutoff = time.time() - RESULT_TTL_SECONDS
        with self.lock:
            expired = [
                job_id
                for job_id, job in self.jobs.items()
                if float(job["created_at"]) < cutoff
                and job["status"] not in {"queued", "processing"}
            ]
            for job_id in expired:
                self.jobs.pop(job_id, None)


def classify_download_failure(exc: BaseException) -> str:
    """Classify common yt-dlp/X failures for PoC diagnostics."""
    details = " ".join(str(item) for item in (exc, exc.__cause__) if item).lower()
    if any(marker in details for marker in ("sign in", "login", "cookie", "authentication")):
        return "AUTH_REQUIRED"
    if any(marker in details for marker in ("expired", "ended", "deleted", "not available")):
        return "EXPIRED_SPACE"
    if any(
        marker in details
        for marker in ("unsupported url", "extractor", "no suitable extractor")
    ):
        return "EXTRACTOR_ERROR"
    if any(marker in details for marker in ("403", "429", "access denied", "forbidden")):
        return "BLOCKED"
    if any(
        marker in details
        for marker in ("timeout", "timed out", "dns", "connection", "network")
    ):
        return "NETWORK_ERROR"
    return "CODE_ERROR"


manager = JobManager()
app = FastAPI(title="X Space Translator Cloud Worker PoC", version="0.2.0-poc")


@app.middleware("http")
async def limit_json_body(request: Request, call_next):  # type: ignore[no-untyped-def]
    if request.method == "POST" and request.url.path in {"/jobs", "/transcribe"}:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_JSON_BODY_BYTES:
            return JSONResponse(
                {"detail": "Request body too large"},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        if len(await request.body()) > MAX_JSON_BODY_BYTES:
            return JSONResponse(
                {"detail": "Request body too large"},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
    return await call_next(request)


def require_worker_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("WORKER_API_KEY", "")
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "WORKER_API_KEY is not configured",
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def submit_url(payload: TranscribeRequest) -> dict[str, object]:
    try:
        return manager.submit_url(payload.url)
    except BusyError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "x-space-translator-worker", "status": "ok"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "ffmpeg": bool(find_ffmpeg()),
        "yt_dlp": importlib.util.find_spec("yt_dlp") is not None,
        "whisper": importlib.util.find_spec("faster_whisper") is not None,
    }


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_job(
    payload: TranscribeRequest,
    _: Annotated[None, Depends(require_worker_api_key)],
) -> dict[str, object]:
    return submit_url(payload)


@app.post("/transcribe", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
def transcribe_compatibility_alias(
    payload: TranscribeRequest,
    _: Annotated[None, Depends(require_worker_api_key)],
) -> dict[str, object]:
    return submit_url(payload)


def accept_upload(file: UploadFile) -> dict[str, object]:
    try:
        extension = validate_upload(file.filename or "", file.content_type)
    except AppError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    try:
        job = manager.reserve_file(file.filename or f"upload{extension}")
    except BusyError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc

    job_id = str(job["job_id"])
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    source = job_dir / f"upload{extension}"
    total = 0
    try:
        with source.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        "Upload too large",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")
        return manager.dispatch_file(job, source)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        manager.abandon(job_id)
        raise


@app.post("/jobs/file", status_code=status.HTTP_202_ACCEPTED)
def create_file_job(
    file: Annotated[UploadFile, File()],
    _: Annotated[None, Depends(require_worker_api_key)],
) -> dict[str, object]:
    return accept_upload(file)


@app.post(
    "/transcribe/file",
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
def transcribe_file_compatibility_alias(
    file: Annotated[UploadFile, File()],
    _: Annotated[None, Depends(require_worker_api_key)],
) -> dict[str, object]:
    return accept_upload(file)


def find_job(job_id: str) -> dict[str, object]:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    _: Annotated[None, Depends(require_worker_api_key)],
) -> dict[str, object]:
    return manager.status_view(find_job(job_id))


@app.get("/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    _: Annotated[None, Depends(require_worker_api_key)],
) -> dict[str, object]:
    return manager.result_view(find_job(job_id))
