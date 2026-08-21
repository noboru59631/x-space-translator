"""FastAPI job service for X Space media acquisition and Groq transcription."""

from __future__ import annotations

import hmac
import ipaddress
import importlib.util
import logging
import os
import shutil
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import psutil
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.downloader import download_space, normalize_x_url
from app.services.errors import AppError, DownloadError
from render_bridge_app.errors import BridgeError
from render_bridge_app.groq_client import transcribe_m4a
from render_bridge_app.media import dependencies_available, probe_m4a, remux_to_m4a
from render_bridge_app.translation_jobs import (
    TranslationBusyError,
    TranslationJobManager,
)

LOGGER = logging.getLogger(__name__)
WORK_ROOT = Path(
    os.getenv("BRIDGE_TEMP_DIR", "/tmp/x-space-translator-render-bridge")
)
WORK_ROOT.mkdir(parents=True, exist_ok=True)
MAX_JSON_BODY_BYTES = 8192
MAX_TRANSLATION_JSON_BODY_BYTES = max(
    65536,
    int(os.getenv("MAX_TRANSLATION_JSON_BODY_BYTES", "1048576")),
)
JOB_TTL_SECONDS = max(
    600,
    int(os.getenv("JOB_TTL_SECONDS", os.getenv("RESULT_TTL_SECONDS", "1800"))),
)
GROQ_FREE_MAX_MB = 25
PUBLIC_RATE_LIMIT_JOBS = max(1, int(os.getenv("PUBLIC_RATE_LIMIT_JOBS", "2")))
PUBLIC_RATE_LIMIT_WINDOW_SECONDS = max(
    60,
    int(os.getenv("PUBLIC_RATE_LIMIT_WINDOW_SECONDS", "600")),
)
PUBLIC_MAX_AUDIO_SECONDS = max(
    60,
    int(os.getenv("PUBLIC_MAX_AUDIO_SECONDS", "7200")),
)
PUBLIC_TRANSLATION_RATE_LIMIT_JOBS = max(
    1,
    int(os.getenv("PUBLIC_TRANSLATION_RATE_LIMIT_JOBS", "2")),
)
PUBLIC_TRANSLATION_RATE_LIMIT_WINDOW_SECONDS = max(
    60,
    int(os.getenv("PUBLIC_TRANSLATION_RATE_LIMIT_WINDOW_SECONDS", "600")),
)
PUBLIC_TRANSLATION_MAX_SEGMENTS = max(
    1,
    int(os.getenv("PUBLIC_TRANSLATION_MAX_SEGMENTS", "500")),
)
PUBLIC_TRANSLATION_MAX_CHARACTERS = max(
    1000,
    int(os.getenv("PUBLIC_TRANSLATION_MAX_CHARACTERS", "120000")),
)
PUBLIC_TRANSLATION_BATCH_SEGMENTS = max(
    1,
    int(os.getenv("PUBLIC_TRANSLATION_BATCH_SEGMENTS", "30")),
)
PUBLIC_TRANSLATION_BATCH_CHARACTERS = max(
    1000,
    int(os.getenv("PUBLIC_TRANSLATION_BATCH_CHARACTERS", "6000")),
)


class BusyError(RuntimeError):
    """Only one download/transcription job can run on a free instance."""


class RateLimitError(RuntimeError):
    """A public client has exhausted its job creation allowance."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("Public job rate limit exceeded")
        self.retry_after = retry_after


class IpRateLimiter:
    """Small in-memory fixed-window limiter for the public PoC endpoint."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def consume(self, client_ip: str) -> float:
        now = time.time()
        cutoff = now - self.window_seconds
        with self.lock:
            events = self.events.setdefault(client_ip, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(events[0] + self.window_seconds - now) + 1)
                raise RateLimitError(retry_after)
            events.append(now)
        return now

    def refund(self, client_ip: str, event: float) -> None:
        with self.lock:
            events = self.events.get(client_ip)
            if not events:
                return
            try:
                events.remove(event)
            except ValueError:
                return
            if not events:
                self.events.pop(client_ip, None)

    def clear(self) -> None:
        with self.lock:
            self.events.clear()


class TranscribeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        try:
            return normalize_x_url(value)
        except AppError as exc:
            raise ValueError(str(exc)) from exc


class TranslationSegment(BaseModel):
    speaker: str = Field(default="Speaker", min_length=1, max_length=100)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    original: str = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def validate_timestamps(self) -> TranslationSegment:
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        self.original = self.original.strip()
        if not self.original:
            raise ValueError("original must not be blank")
        return self


class TranslationRequest(BaseModel):
    segments: list[TranslationSegment] = Field(
        min_length=1,
        max_length=PUBLIC_TRANSLATION_MAX_SEGMENTS,
    )

    @model_validator(mode="after")
    def validate_total_characters(self) -> TranslationRequest:
        if sum(len(segment.original) for segment in self.segments) > (
            PUBLIC_TRANSLATION_MAX_CHARACTERS
        ):
            raise ValueError("Transcript exceeds the public translation character limit")
        return self


def groq_upload_limit_bytes() -> int:
    requested = max(1, int(os.getenv("GROQ_MAX_UPLOAD_MB", "25")))
    return min(GROQ_FREE_MAX_MB, requested) * 1024 * 1024


class JobManager:
    """Single-worker, in-memory PoC job manager."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()
        self.active_job_id: str | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="render-bridge",
        )

    def is_busy(self) -> bool:
        with self.lock:
            return self.active_job_id is not None

    def submit(
        self,
        source_url: str,
        *,
        visibility: str = "authenticated",
    ) -> dict[str, object]:
        job = self._reserve(source_url, visibility=visibility)
        queued_view = self.status_view(job)
        try:
            self.executor.submit(
                self._run,
                str(job["job_id"]),
                source_url,
                visibility,
            )
        except Exception:
            self._abandon(str(job["job_id"]))
            raise
        return queued_view

    def get(self, job_id: str) -> dict[str, object] | None:
        self._expire()
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    @staticmethod
    def status_view(job: dict[str, object]) -> dict[str, object]:
        fields = (
            "job_id",
            "status",
            "stage",
            "progress",
            "error_code",
            "error",
            "media_duration",
            "media_size_bytes",
            "media_codec",
            "download_seconds",
            "remux_seconds",
            "groq_seconds",
            "elapsed_seconds",
            "peak_memory_mb",
            "cleanup",
        )
        return {field: job.get(field) for field in fields}

    @classmethod
    def result_view(cls, job: dict[str, object]) -> dict[str, object]:
        result = job.get("result")
        if job.get("status") == "completed" and isinstance(result, dict):
            return dict(result)
        return cls.status_view(job)

    @staticmethod
    def api_status_view(job: dict[str, object]) -> dict[str, object]:
        view: dict[str, object] = {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "progress": None,
        }
        if job.get("status") == "failed":
            view["error_code"] = job.get("public_error_code") or "INTERNAL_ERROR"
        return view

    @classmethod
    def api_result_view(cls, job: dict[str, object]) -> dict[str, object]:
        status_value = str(job["status"])
        view: dict[str, object] = {
            "job_id": job["job_id"],
            "status": status_value,
        }
        if status_value == "completed" and isinstance(job.get("result"), dict):
            view["result"] = dict(job["result"])  # type: ignore[arg-type]
        elif status_value == "failed":
            view["error_code"] = job.get("public_error_code") or "INTERNAL_ERROR"
        return view

    def _reserve(
        self,
        source_url: str,
        *,
        visibility: str = "authenticated",
    ) -> dict[str, object]:
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": None,
            "source_url": source_url,
            "visibility": visibility,
            "created_at": time.time(),
            "error_code": None,
            "public_error_code": None,
            "error": None,
            "media_duration": None,
            "media_size_bytes": None,
            "media_codec": None,
            "download_seconds": None,
            "remux_seconds": None,
            "groq_seconds": None,
            "elapsed_seconds": None,
            "peak_memory_mb": None,
            "cleanup": False,
            "result": None,
            "finished_at": None,
        }
        with self.lock:
            if self.active_job_id is not None:
                raise BusyError("Bridge is already processing another job")
            self.active_job_id = job_id
            self.jobs[job_id] = job
        return job

    def _update(self, job_id: str, **values: object) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(values)

    def _abandon(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id, None)
            if self.active_job_id == job_id:
                self.active_job_id = None

    def _finish(self, job_id: str) -> None:
        with self.lock:
            if self.active_job_id == job_id:
                self.active_job_id = None

    def _run(self, job_id: str, source_url: str, visibility: str) -> None:
        job_dir = WORK_ROOT / job_id
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        result: dict[str, object] | None = None
        failure: tuple[str, str] | None = None
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
            download_started = time.perf_counter()
            self._update(
                job_id,
                status="processing",
                stage="downloading",
                progress=None,
            )
            source, metadata = download_space(source_url, job_dir)
            self._update(
                job_id,
                download_seconds=round(time.perf_counter() - download_started, 3),
            )

            remux_started = time.perf_counter()
            self._update(job_id, stage="validating_audio", progress=None)
            m4a_path = remux_to_m4a(source, job_dir / "groq-upload.m4a")
            media = probe_m4a(m4a_path)
            self._update(
                job_id,
                media_duration=round(media.duration, 3),
                media_size_bytes=media.size_bytes,
                media_codec=media.codec,
                remux_seconds=round(time.perf_counter() - remux_started, 3),
                progress=None,
            )
            if (
                visibility == "public"
                and media.duration > PUBLIC_MAX_AUDIO_SECONDS
            ):
                raise BridgeError(
                    "AUDIO_TOO_LONG",
                    "The X Space exceeds the public API duration limit",
                )
            if media.size_bytes > groq_upload_limit_bytes():
                raise BridgeError(
                    "GROQ_FILE_TOO_LARGE",
                    "The validated M4A exceeds the Groq Free Tier 25 MB limit",
                )

            groq_started = time.perf_counter()
            self._update(job_id, stage="transcribing", progress=None)
            transcript = transcribe_m4a(
                m4a_path,
                os.environ["GROQ_API_KEY"],
            )
            self._update(
                job_id,
                groq_seconds=round(time.perf_counter() - groq_started, 3),
                progress=None,
            )
            self._update(job_id, stage="preparing_result", progress=None)
            result = {
                "title": str(metadata.get("title") or ""),
                "source_url": source_url,
                "detected_language": transcript["detected_language"],
                "duration": transcript["duration"] or round(media.duration, 3),
                "segments": transcript["segments"],
            }
        except DownloadError as exc:
            failure = (classify_download_failure(exc), str(exc))
        except BridgeError as exc:
            failure = (exc.code, str(exc))
        except Exception:
            LOGGER.exception("Unexpected render bridge job failure: %s", job_id)
            failure = ("CODE_ERROR", "Unexpected bridge error")
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            stop_sampling.set()
            common = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_memory_mb": round(peak[0], 1),
                "cleanup": not job_dir.exists(),
                "progress": None,
                "finished_at": time.time(),
            }
            if result is not None:
                self._update(
                    job_id,
                    **common,
                    status="completed",
                    stage="completed",
                    result=result,
                )
            else:
                code, message = failure or ("CODE_ERROR", "Unexpected bridge error")
                self._update(
                    job_id,
                    **common,
                    status="failed",
                    stage="failed",
                    error_code=code,
                    public_error_code=public_error_code(code),
                    error=message,
                )
            self._finish(job_id)

    @staticmethod
    def _start_memory_sampler() -> tuple[threading.Event, list[float]]:
        stopped = threading.Event()
        process = psutil.Process()
        peak = [process.memory_info().rss / 1024 / 1024]

        def sample() -> None:
            while not stopped.wait(0.25):
                try:
                    rss = process.memory_info().rss
                    rss += sum(
                        child.memory_info().rss
                        for child in process.children(recursive=True)
                    )
                    peak[0] = max(peak[0], rss / 1024 / 1024)
                except (OSError, psutil.Error):
                    pass

        threading.Thread(
            target=sample,
            daemon=True,
            name="render-bridge-memory",
        ).start()
        return stopped, peak

    def _expire(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        with self.lock:
            expired = [
                job_id
                for job_id, job in self.jobs.items()
                if float(job.get("finished_at") or job["created_at"]) < cutoff
                and job["status"] not in {"queued", "processing"}
            ]
            for job_id in expired:
                self.jobs.pop(job_id, None)


def classify_download_failure(exc: BaseException) -> str:
    details = " ".join(str(item) for item in (exc, exc.__cause__) if item).lower()
    if any(marker in details for marker in ("sign in", "login", "cookie", "auth")):
        return "AUTH_REQUIRED"
    if any(marker in details for marker in ("403", "429", "forbidden", "blocked")):
        return "BLOCKED"
    if any(marker in details for marker in ("extractor", "unsupported url")):
        return "EXTRACTOR_ERROR"
    if any(
        marker in details
        for marker in ("timeout", "timed out", "dns", "connection", "network")
    ):
        return "NETWORK_ERROR"
    return "DOWNLOAD_ERROR"


def public_error_code(internal_code: str) -> str:
    """Map internal diagnostics to a small, stable Bankr-facing code set."""
    if internal_code in {
        "AUTH_REQUIRED",
        "BLOCKED",
        "EXTRACTOR_ERROR",
        "NETWORK_ERROR",
        "DOWNLOAD_ERROR",
    }:
        return "X_DOWNLOAD_FAILED"
    if internal_code.startswith("GROQ_"):
        return "GROQ_TRANSCRIPTION_FAILED"
    if internal_code in {
        "INVALID_MEDIA",
        "INVALID_CONTAINER",
        "INVALID_AUDIO_CODEC",
        "INVALID_DURATION",
        "REMUX_FAILED",
        "FFMPEG_MISSING",
        "FFPROBE_MISSING",
        "AUDIO_TOO_LONG",
    }:
        return (
            "AUDIO_TOO_LONG"
            if internal_code == "AUDIO_TOO_LONG"
            else "AUDIO_INVALID"
        )
    return "INTERNAL_ERROR"


manager = JobManager()
public_rate_limiter = IpRateLimiter(
    PUBLIC_RATE_LIMIT_JOBS,
    PUBLIC_RATE_LIMIT_WINDOW_SECONDS,
)
public_translation_rate_limiter = IpRateLimiter(
    PUBLIC_TRANSLATION_RATE_LIMIT_JOBS,
    PUBLIC_TRANSLATION_RATE_LIMIT_WINDOW_SECONDS,
)
translation_manager = TranslationJobManager(
    ttl_seconds=JOB_TTL_SECONDS,
    batch_segments=PUBLIC_TRANSLATION_BATCH_SEGMENTS,
    batch_characters=PUBLIC_TRANSLATION_BATCH_CHARACTERS,
)
operation_submission_lock = threading.Lock()
app = FastAPI(title="X Space Translator Render Bridge PoC", version="0.1.0-poc")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    if request.url.path in {
        "/api/jobs",
        "/public/jobs",
        "/public/translations",
    }:
        return JSONResponse(
            {
                "error_code": (
                    "INVALID_TRANSLATION_REQUEST"
                    if request.url.path == "/public/translations"
                    else "INVALID_URL"
                )
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def limit_json_body(request: Request, call_next):  # type: ignore[no-untyped-def]
    limited_paths = {
        "/jobs",
        "/transcribe",
        "/api/jobs",
        "/public/jobs",
        "/public/translations",
    }
    if request.method == "POST" and request.url.path in limited_paths:
        body_limit = (
            MAX_TRANSLATION_JSON_BODY_BYTES
            if request.url.path == "/public/translations"
            else MAX_JSON_BODY_BYTES
        )
        content_length = request.headers.get("content-length")
        try:
            declared_length = int(content_length) if content_length else 0
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if declared_length > body_limit:
            return JSONResponse(
                {"detail": "Request body too large"},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        if len(await request.body()) > body_limit:
            return JSONResponse(
                {"detail": "Request body too large"},
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
    return await call_next(request)


def require_bridge_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("BRIDGE_API_KEY", "")
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "BRIDGE_API_KEY is not configured",
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def submit_job(
    payload: TranscribeRequest,
    *,
    busy_detail: str | None = None,
    visibility: str = "authenticated",
) -> dict[str, object]:
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "GROQ_API_KEY is not configured",
        )
    try:
        with operation_submission_lock:
            if translation_manager.is_busy():
                raise BusyError("Bridge is already processing another job")
            return manager.submit(payload.url, visibility=visibility)
    except BusyError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            busy_detail or str(exc),
        ) from exc


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "x-space-translator-render-bridge", "status": "ok"}


@app.get("/health")
def health() -> dict[str, object]:
    ffmpeg, ffprobe = dependencies_available()
    return {
        "status": "ok",
        "ffmpeg": ffmpeg and ffprobe,
        "yt_dlp": importlib.util.find_spec("yt_dlp") is not None,
    }


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_job(
    payload: TranscribeRequest,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    return submit_job(payload)


@app.post("/transcribe", status_code=status.HTTP_202_ACCEPTED)
def transcribe(
    payload: TranscribeRequest,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    return submit_job(payload)


@app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_api_job(
    payload: TranscribeRequest,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    job = submit_job(payload, busy_detail="BUSY")
    return {"job_id": job["job_id"], "status": job["status"]}


def public_client_ip(request: Request) -> str:
    """Return Render's first forwarded client address, failing closed."""
    forwarded = request.headers.get("x-forwarded-for", "")
    candidate = forwarded.split(",", 1)[0].strip() if forwarded else ""
    if not candidate and request.client:
        candidate = request.client.host
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return "unknown"


@app.post("/public/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_public_job(
    payload: TranscribeRequest,
    request: Request,
) -> dict[str, object]:
    client_ip = public_client_ip(request)
    try:
        rate_event = public_rate_limiter.consume(client_ip)
    except RateLimitError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "RATE_LIMITED",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    try:
        job = submit_job(
            payload,
            busy_detail="BUSY",
            visibility="public",
        )
    except Exception:
        public_rate_limiter.refund(client_ip, rate_event)
        raise
    return {"job_id": job["job_id"], "status": job["status"]}


@app.post("/public/translations", status_code=status.HTTP_202_ACCEPTED)
def create_public_translation(
    payload: TranslationRequest,
    request: Request,
) -> dict[str, object]:
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "GROQ_API_KEY is not configured",
        )
    client_ip = public_client_ip(request)
    try:
        rate_event = public_translation_rate_limiter.consume(client_ip)
    except RateLimitError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "RATE_LIMITED",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    try:
        with operation_submission_lock:
            if manager.is_busy():
                raise TranslationBusyError("Bridge is already processing another job")
            job = translation_manager.submit(
                [segment.model_dump() for segment in payload.segments]
            )
    except TranslationBusyError as exc:
        public_translation_rate_limiter.refund(client_ip, rate_event)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "BUSY",
        ) from exc
    except Exception:
        public_translation_rate_limiter.refund(client_ip, rate_event)
        raise
    return {"job_id": job["job_id"], "status": job["status"]}


def find_job(job_id: str) -> dict[str, object]:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


def find_public_job(job_id: str) -> dict[str, object]:
    job = find_job(job_id)
    if job.get("visibility") != "public":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


def find_translation_job(job_id: str) -> dict[str, object]:
    job = translation_manager.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    return manager.status_view(find_job(job_id))


@app.get("/jobs/{job_id}/result")
def get_result(
    job_id: str,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    return manager.result_view(find_job(job_id))


@app.get("/api/jobs/{job_id}")
def get_api_job(
    job_id: str,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    return manager.api_status_view(find_job(job_id))


@app.get("/api/jobs/{job_id}/result")
def get_api_result(
    job_id: str,
    _: Annotated[None, Depends(require_bridge_key)],
) -> dict[str, object]:
    return manager.api_result_view(find_job(job_id))


@app.get("/public/jobs/{job_id}")
def get_public_job(job_id: str) -> dict[str, object]:
    return manager.api_status_view(find_public_job(job_id))


@app.get("/public/jobs/{job_id}/result")
def get_public_result(job_id: str) -> dict[str, object]:
    return manager.api_result_view(find_public_job(job_id))


@app.get("/public/translations/{translation_job_id}")
def get_public_translation(translation_job_id: str) -> dict[str, object]:
    return translation_manager.status_view(find_translation_job(translation_job_id))


@app.get("/public/translations/{translation_job_id}/result")
def get_public_translation_result(
    translation_job_id: str,
) -> dict[str, object]:
    return translation_manager.result_view(find_translation_job(translation_job_id))
