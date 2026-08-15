"""Asynchronous cloud transcription proof-of-concept worker."""

from __future__ import annotations

import hmac
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
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, field_validator

from app.services.audio import convert_to_wav, find_ffmpeg, validate_upload
from app.services.downloader import download_space, normalize_x_url
from app.services.errors import AppError, DownloadError
from app.services.transcriber import Transcriber

WORK_ROOT = Path(os.getenv("WORKER_TEMP_DIR", "/tmp/x-space-worker"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "256"))
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", "3600"))
LOGGER = logging.getLogger(__name__)


class TranscribeRequest(BaseModel):
    url: str

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
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cloud-worker")
        self.transcriber = Transcriber()

    def submit_url(self, url: str) -> dict[str, object]:
        job = self._new_job("url", url)
        self.executor.submit(self._run_url, str(job["job_id"]), url)
        return self.public(job)

    def submit_file(self, source: Path) -> dict[str, object]:
        job = self._new_job("file", source.name)
        self.executor.submit(self._run_file, str(job["job_id"]), source)
        return self.public(job)

    def get(self, job_id: str) -> dict[str, object] | None:
        self._expire_old_jobs()
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    @staticmethod
    def public(job: dict[str, object]) -> dict[str, object]:
        data = {
            key: job.get(key)
            for key in (
                "job_id",
                "status",
                "stage",
                "error_code",
                "error",
                "elapsed_seconds",
                "peak_memory_mb",
                "audio_deleted",
            )
        }
        if job.get("status") == "completed":
            result = job.get("result")
            if isinstance(result, dict):
                data.update(result)
        return data

    def _new_job(self, source_type: str, source: str) -> dict[str, object]:
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
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
            self.jobs[job_id] = job
        return job

    def _update(self, job_id: str, **values: object) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(values)

    def _run_url(self, job_id: str, url: str) -> None:
        job_dir = WORK_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        try:
            self._update(job_id, status="processing", stage="downloading")
            source, _ = download_space(url, job_dir)
            self._transcribe(job_id, source, job_dir)
        except DownloadError as exc:
            self._fail(job_id, classify_download_failure(exc), str(exc))
        except AppError as exc:
            self._fail(job_id, "code_error", str(exc))
        except Exception:
            LOGGER.exception("Unexpected URL job failure: %s", job_id)
            self._fail(job_id, "code_error", "Unexpected worker error")
        finally:
            stop_sampling.set()
            shutil.rmtree(job_dir, ignore_errors=True)
            self._update(
                job_id,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                peak_memory_mb=round(peak[0], 1),
                audio_deleted=not job_dir.exists(),
            )

    def _run_file(self, job_id: str, source: Path) -> None:
        job_dir = source.parent
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        try:
            self._update(job_id, status="processing", stage="converting")
            self._transcribe(job_id, source, job_dir)
        except AppError as exc:
            self._fail(job_id, "code_error", str(exc))
        except Exception:
            LOGGER.exception("Unexpected upload job failure: %s", job_id)
            self._fail(job_id, "code_error", "Unexpected worker error")
        finally:
            stop_sampling.set()
            shutil.rmtree(job_dir, ignore_errors=True)
            self._update(
                job_id,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                peak_memory_mb=round(peak[0], 1),
                audio_deleted=not job_dir.exists(),
            )

    def _transcribe(self, job_id: str, source: Path, job_dir: Path) -> None:
        self._update(job_id, status="processing", stage="converting")
        wav_path = convert_to_wav(source, job_dir / "audio.wav")
        self._update(job_id, stage="transcribing")
        raw = self.transcriber.transcribe(
            wav_path,
            mode="light",
            device_preference="cpu",
            model_override="base",
            compute_override="int8",
        )
        result = {
            "status": "completed",
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
        self._update(job_id, status="completed", stage="completed", result=result)

    def _fail(self, job_id: str, code: str, message: str) -> None:
        self._update(job_id, status="failed", stage="failed", error_code=code, error=message)

    @staticmethod
    def _start_memory_sampler() -> tuple[threading.Event, list[float]]:
        stopped = threading.Event()
        peak = [0.0]
        process = psutil.Process()

        def sample() -> None:
            while not stopped.wait(0.25):
                try:
                    rss = process.memory_info().rss
                    rss += sum(child.memory_info().rss for child in process.children(recursive=True))
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
                if float(job["created_at"]) < cutoff and job["status"] != "processing"
            ]
            for job_id in expired:
                self.jobs.pop(job_id, None)


def classify_download_failure(exc: BaseException) -> str:
    """Classify common yt-dlp/X failures for PoC diagnostics."""
    details = " ".join(str(item) for item in (exc, exc.__cause__) if item).lower()
    if any(marker in details for marker in ("sign in", "login", "cookie", "authentication")):
        return "authentication_required"
    if any(marker in details for marker in ("expired", "ended", "deleted", "not available")):
        return "expired_space"
    if any(marker in details for marker in ("unsupported url", "extractor", "no suitable extractor")):
        return "yt_dlp_extractor"
    if any(marker in details for marker in ("403", "429", "access denied", "forbidden")):
        return "datacenter_ip_block"
    if any(marker in details for marker in ("timeout", "timed out", "dns", "connection", "network")):
        return "network"
    return "code_error"


manager = JobManager()
app = FastAPI(title="X Space Translator Cloud Worker PoC", version="0.1.0-poc")


def require_worker_token(
    x_worker_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("WORKER_API_TOKEN", "")
    if expected and not (x_worker_token and hmac.compare_digest(x_worker_token, expected)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid worker token")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "x-space-translator-worker", "status": "ok"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "ffmpeg": bool(find_ffmpeg()),
        "model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "persistent_storage": False,
    }


@app.post("/transcribe", status_code=status.HTTP_202_ACCEPTED)
def transcribe(
    payload: TranscribeRequest,
    _: Annotated[None, Depends(require_worker_token)],
) -> dict[str, object]:
    return manager.submit_url(payload.url)


@app.post("/transcribe/file", status_code=status.HTTP_202_ACCEPTED)
def transcribe_file(
    file: Annotated[UploadFile, File()],
    _: Annotated[None, Depends(require_worker_token)],
) -> dict[str, object]:
    try:
        extension = validate_upload(file.filename or "", file.content_type)
    except AppError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    job_id = uuid.uuid4().hex
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    source = job_dir / f"upload{extension}"
    total = 0
    try:
        with source.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Upload too large")
                output.write(chunk)
        if total == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")
        return manager.submit_file(source)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    _: Annotated[None, Depends(require_worker_token)],
) -> dict[str, object]:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return manager.public(job)
