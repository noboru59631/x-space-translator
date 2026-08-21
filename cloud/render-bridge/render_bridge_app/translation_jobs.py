"""Single-worker, in-memory public translation job pipeline."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import psutil

from render_bridge_app.errors import BridgeError
from render_bridge_app.translation_client import translate_batch
from render_bridge_app.translation_validation import validate_preservation

LOGGER = logging.getLogger(__name__)


def public_translation_error_code(internal_code: str) -> str:
    """Expose a safe operational class without leaking provider details."""
    if internal_code == "GROQ_TRANSLATION_RATE_LIMIT":
        return "GROQ_RATE_LIMITED"
    if internal_code == "GROQ_TRANSLATION_ALIGNMENT":
        return "TRANSLATION_ALIGNMENT_FAILED"
    if internal_code == "GROQ_TRANSLATION_MISSING":
        return "TRANSLATION_MISSING"
    if internal_code in {
        "GROQ_TRANSLATION_INVALID_RESPONSE",
        "GROQ_TRANSLATION_RESPONSE_TOO_LARGE",
    }:
        return "TRANSLATION_INVALID"
    if internal_code.startswith("GROQ_"):
        return "GROQ_TRANSLATION_FAILED"
    return "INTERNAL_ERROR"


class TranslationBusyError(RuntimeError):
    """Only one translation job can run at a time."""


class TranslationJobManager:
    def __init__(
        self,
        *,
        ttl_seconds: int,
        batch_segments: int,
        batch_characters: int,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.batch_segments = batch_segments
        self.batch_characters = batch_characters
        self.jobs: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()
        self.active_job_id: str | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="render-translation",
        )

    def is_busy(self) -> bool:
        with self.lock:
            return self.active_job_id is not None

    def submit(self, segments: list[dict[str, object]]) -> dict[str, object]:
        job = self._reserve()
        try:
            self.executor.submit(self._run, str(job["job_id"]), segments)
        except Exception:
            self._abandon(str(job["job_id"]))
            raise
        return self.status_view(job)

    def get(self, job_id: str) -> dict[str, object] | None:
        self._expire()
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    @staticmethod
    def status_view(job: dict[str, object]) -> dict[str, object]:
        view: dict[str, object] = {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "progress": None,
        }
        if job["status"] == "failed":
            view["error_code"] = job.get("public_error_code") or "INTERNAL_ERROR"
        return view

    @classmethod
    def result_view(cls, job: dict[str, object]) -> dict[str, object]:
        view: dict[str, object] = {
            "job_id": job["job_id"],
            "status": job["status"],
        }
        if job["status"] == "completed" and isinstance(job.get("result"), dict):
            view["result"] = dict(job["result"])  # type: ignore[arg-type]
        elif job["status"] == "failed":
            view["error_code"] = job.get("public_error_code") or "INTERNAL_ERROR"
        return view

    def diagnostic_view(self, job: dict[str, object]) -> dict[str, object]:
        return {
            "elapsed_seconds": job.get("elapsed_seconds"),
            "peak_memory_mb": job.get("peak_memory_mb"),
            "cleanup": job.get("cleanup"),
        }

    def _reserve(self) -> dict[str, object]:
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "created_at": time.time(),
            "finished_at": None,
            "public_error_code": None,
            "result": None,
            "elapsed_seconds": None,
            "peak_memory_mb": None,
            "cleanup": False,
        }
        with self.lock:
            if self.active_job_id is not None:
                raise TranslationBusyError("Translation is already processing")
            self.active_job_id = job_id
            self.jobs[job_id] = job
        return job

    def _abandon(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id, None)
            if self.active_job_id == job_id:
                self.active_job_id = None

    def _update(self, job_id: str, **values: object) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(values)

    def _finish(self, job_id: str) -> None:
        with self.lock:
            if self.active_job_id == job_id:
                self.active_job_id = None

    def _run(self, job_id: str, segments: list[dict[str, object]]) -> None:
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        result: dict[str, object] | None = None
        public_error = "INTERNAL_ERROR"
        try:
            self._update(job_id, status="processing", stage="translating")
            indexed = [
                {"id": index, "text": str(segment["original"])}
                for index, segment in enumerate(segments)
            ]
            translations: dict[int, str] = {}
            for batch in self._batches(indexed):
                translations.update(self._translate_aligned(batch))

            checks_before = [
                validate_preservation(
                    str(segment["original"]),
                    translations[index],
                )
                for index, segment in enumerate(segments)
            ]
            retry_ids = [
                index for index, check in enumerate(checks_before) if not check.ok
            ]
            if retry_ids:
                self._update(job_id, stage="retrying_preservation")
                retry_items = [indexed[index] for index in retry_ids]
                for batch in self._batches(retry_items):
                    translations.update(self._translate_aligned(batch, retry=True))

            checks_after = [
                validate_preservation(
                    str(segment["original"]),
                    translations[index],
                )
                for index, segment in enumerate(segments)
            ]
            output_segments = [
                {
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "original": segment["original"],
                    "translation": translations[index],
                    "translation_warning": not checks_after[index].ok,
                }
                for index, segment in enumerate(segments)
            ]
            result = {
                "segments": output_segments,
                "translated_segments": len(output_segments),
                "missing_translations": sum(
                    not str(segment["translation"]).strip()
                    for segment in output_segments
                ),
                "alignment": len(output_segments) == len(segments),
                "warnings_before_retry": len(retry_ids),
                "number_warnings_before_retry": sum(
                    not check.number_ok for check in checks_before
                ),
                "number_warnings_after_retry": sum(
                    not check.number_ok for check in checks_after
                ),
                "ticker_warnings_before_retry": sum(
                    not check.ticker_ok for check in checks_before
                ),
                "ticker_warnings_after_retry": sum(
                    not check.ticker_ok for check in checks_after
                ),
                "url_warnings_before_retry": sum(
                    not check.url_ok for check in checks_before
                ),
                "url_warnings_after_retry": sum(
                    not check.url_ok for check in checks_after
                ),
                "remaining_warnings": sum(not check.ok for check in checks_after),
            }
        except BridgeError as exc:
            public_error = public_translation_error_code(exc.code)
        except Exception:
            LOGGER.exception("Unexpected translation job failure: %s", job_id)
        finally:
            stop_sampling.set()
            common = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_memory_mb": round(peak[0], 1),
                "cleanup": True,
                "finished_at": time.time(),
            }
            if result is not None:
                result.update(common)
                self._update(
                    job_id,
                    **common,
                    status="completed",
                    stage="completed",
                    result=result,
                )
            else:
                self._update(
                    job_id,
                    **common,
                    status="failed",
                    stage="failed",
                    public_error_code=public_error,
                )
            self._finish(job_id)

    def _batches(
        self,
        items: list[dict[str, object]],
    ) -> list[list[dict[str, object]]]:
        batches: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        current_characters = 0
        for item in items:
            characters = len(str(item["text"]))
            if current and (
                len(current) >= self.batch_segments
                or current_characters + characters > self.batch_characters
            ):
                batches.append(current)
                current = []
                current_characters = 0
            current.append(item)
            current_characters += characters
        if current:
            batches.append(current)
        return batches

    def _translate_aligned(
        self,
        items: list[dict[str, object]],
        *,
        retry: bool = False,
    ) -> dict[int, str]:
        try:
            return translate_batch(
                items,
                os.environ["GROQ_API_KEY"],
                retry=retry,
            )
        except BridgeError as exc:
            splittable_codes = {
                "GROQ_TRANSLATION_INVALID_RESPONSE",
                "GROQ_TRANSLATION_ALIGNMENT",
                "GROQ_TRANSLATION_MISSING",
                "GROQ_TRANSLATION_RESPONSE_TOO_LARGE",
            }
            if exc.code not in splittable_codes or len(items) <= 1:
                raise
            midpoint = len(items) // 2
            combined = self._translate_aligned(items[:midpoint], retry=retry)
            combined.update(self._translate_aligned(items[midpoint:], retry=retry))
            return combined

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self.lock:
            expired = [
                job_id
                for job_id, job in self.jobs.items()
                if float(job.get("finished_at") or job["created_at"]) < cutoff
                and job["status"] not in {"queued", "processing"}
            ]
            for job_id in expired:
                self.jobs.pop(job_id, None)

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
            name="render-translation-memory",
        ).start()
        return stopped, peak
