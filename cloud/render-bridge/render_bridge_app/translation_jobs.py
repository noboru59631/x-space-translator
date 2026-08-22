"""Single-worker, in-memory public translation job pipeline."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor

import psutil

from render_bridge_app.errors import BridgeError
from render_bridge_app.translation_client import translate_batch
from render_bridge_app.translation_rate_limit import (
    DEFAULT_BATCH_TOKEN_TARGET,
    TranslationRateLimitScheduler,
    estimate_batch_tokens,
)
from render_bridge_app.translation_validation import validate_preservation

LOGGER = logging.getLogger(__name__)


def public_translation_error_code(internal_code: str) -> str:
    """Expose a safe operational class without leaking provider details."""
    if internal_code == "GROQ_TRANSLATION_DAILY_LIMIT":
        return "GROQ_DAILY_LIMIT"
    if internal_code == "TRANSLATION_TIMEOUT":
        return "TRANSLATION_TIMEOUT"
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
        batch_token_target: int = DEFAULT_BATCH_TOKEN_TARGET,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.batch_segments = batch_segments
        self.batch_characters = batch_characters
        self.batch_token_target = batch_token_target
        self.jobs: dict[str, dict[str, object]] = {}
        self.lock = threading.Lock()
        self.active_job_id: str | None = None
        self.futures: dict[str, Future[None]] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="render-translation",
        )

    def is_busy(self) -> bool:
        with self.lock:
            return self.active_job_id is not None

    def submit(
        self,
        segments: list[dict[str, object]],
        *,
        result_context: dict[str, object] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        job = self._reserve()
        try:
            future = self.executor.submit(
                self._run,
                str(job["job_id"]),
                segments,
                dict(result_context or {}),
                timeout_seconds,
            )
            with self.lock:
                self.futures[str(job["job_id"])] = future
        except Exception:
            self._abandon(str(job["job_id"]))
            raise
        return self.status_view(job)

    def get(self, job_id: str) -> dict[str, object] | None:
        self._reconcile_future(job_id)
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
        for field in (
            "updated_at",
            "batch_total",
            "batch_completed",
            "current_batch",
            "requests_sent",
            "successful_requests",
            "rate_limit_count",
            "retry_count",
            "waiting",
            "wait_reason",
            "wait_until",
            "retry_after_seconds",
            "last_request_at",
            "last_success_at",
            "last_429_at",
            "rate_limit_wait_count",
            "longest_single_wait_seconds",
        ):
            view[field] = job.get(field)
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
            "updated_at": time.time(),
            "finished_at": None,
            "public_error_code": None,
            "result": None,
            "elapsed_seconds": None,
            "peak_memory_mb": None,
            "cleanup": False,
            "batch_total": 0,
            "batch_completed": 0,
            "current_batch": 0,
            "requests_sent": 0,
            "successful_requests": 0,
            "rate_limit_count": 0,
            "retry_count": 0,
            "waiting": False,
            "wait_reason": None,
            "wait_until": None,
            "retry_after_seconds": 0.0,
            "last_request_at": None,
            "last_success_at": None,
            "last_429_at": None,
            "rate_limit_wait_count": 0,
            "longest_single_wait_seconds": 0.0,
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
            self.futures.pop(job_id, None)
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

    def _run(
        self,
        job_id: str,
        segments: list[dict[str, object]],
        result_context: dict[str, object],
        timeout_seconds: int | None,
    ) -> None:
        started = time.perf_counter()
        stop_sampling, peak = self._start_memory_sampler()
        current_stage = ["translating"]

        def record_telemetry(event: str, values: dict[str, object]) -> None:
            update = dict(values)
            update["updated_at"] = time.time()
            if event == "wait_started":
                update["stage"] = "rate_limit_wait"
            elif event == "wait_finished":
                update["stage"] = current_stage[0]
            self._update(job_id, **update)

        deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        scheduler = TranslationRateLimitScheduler(
            deadline=deadline,
            telemetry=record_telemetry,
        )
        result: dict[str, object] | None = None
        public_error = "INTERNAL_ERROR"
        try:
            self._update(
                job_id,
                status="processing",
                stage="translating",
                updated_at=time.time(),
            )
            indexed = [
                {"id": index, "text": str(segment["original"])}
                for index, segment in enumerate(segments)
            ]
            translations: dict[int, str] = {}
            primary_batches = self._batches(indexed)
            self._update(
                job_id,
                batch_total=len(primary_batches),
                updated_at=time.time(),
            )
            for batch_index, batch in enumerate(primary_batches, start=1):
                scheduler.ensure_time_remaining()
                self._update(
                    job_id,
                    current_batch=batch_index,
                    updated_at=time.time(),
                )
                translations.update(
                    self._translate_aligned(batch, scheduler=scheduler)
                )
                self._update(
                    job_id,
                    batch_completed=batch_index,
                    updated_at=time.time(),
                )

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
                current_stage[0] = "retrying_preservation"
                self._update(
                    job_id,
                    stage=current_stage[0],
                    updated_at=time.time(),
                )
                retry_items = [indexed[index] for index in retry_ids]
                retry_batches = self._batches(retry_items)
                completed_before_retry = len(primary_batches)
                self._update(
                    job_id,
                    batch_total=completed_before_retry + len(retry_batches),
                    updated_at=time.time(),
                )
                for retry_offset, batch in enumerate(retry_batches, start=1):
                    scheduler.ensure_time_remaining()
                    current_batch = completed_before_retry + retry_offset
                    self._update(
                        job_id,
                        current_batch=current_batch,
                        updated_at=time.time(),
                    )
                    translations.update(
                        self._translate_aligned(
                            batch,
                            retry=True,
                            scheduler=scheduler,
                        )
                    )
                    self._update(
                        job_id,
                        batch_completed=current_batch,
                        updated_at=time.time(),
                    )

            checks_after = [
                validate_preservation(
                    str(segment["original"]),
                    translations[index],
                )
                for index, segment in enumerate(segments)
            ]
            output_segments = []
            for index, segment in enumerate(segments):
                output_segment = {
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "original": segment["original"],
                    "translation": translations[index],
                    "translation_warning": not checks_after[index].ok,
                }
                if isinstance(segment.get("index"), int):
                    output_segment["index"] = segment["index"]
                output_segments.append(output_segment)
            result = {
                **result_context,
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
            scheduler_metrics = scheduler.metrics()
            rate_limit_headers = scheduler.safe_headers()
            LOGGER.warning(
                "Translation scheduler job_id=%s batches=%s requests=%s "
                "rate_limit_429=%s wait_seconds=%s avg_segments=%s "
                "avg_characters=%s avg_tokens=%s max_rolling_tokens=%s "
                "prompt_tokens=%s completion_tokens=%s final_429=%s",
                job_id,
                scheduler_metrics["batch_count"],
                scheduler_metrics["requests_sent"],
                scheduler_metrics["rate_limit_429_count"],
                scheduler_metrics["total_wait_seconds"],
                scheduler_metrics["average_segments_per_batch"],
                scheduler_metrics["average_characters_per_batch"],
                scheduler_metrics["average_tokens_per_request"],
                scheduler_metrics["max_rolling_60_second_tokens"],
                scheduler_metrics["prompt_tokens"],
                scheduler_metrics["completion_tokens"],
                scheduler_metrics["final_429_failure"],
            )
            common = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_memory_mb": round(peak[0], 1),
                "cleanup": True,
                "finished_at": time.time(),
                "updated_at": time.time(),
                "waiting": False,
                "wait_reason": None,
                "wait_until": None,
                "retry_after_seconds": 0.0,
            }
            if result is not None:
                result.update(common)
                self._update(
                    job_id,
                    **common,
                    status="completed",
                    stage="completed",
                    result=result,
                    rate_limit_metrics=scheduler_metrics,
                    rate_limit_headers=rate_limit_headers,
                )
            else:
                self._update(
                    job_id,
                    **common,
                    status="failed",
                    stage="failed",
                    public_error_code=public_error,
                    rate_limit_metrics=scheduler_metrics,
                    rate_limit_headers=rate_limit_headers,
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
            candidate = [*current, item]
            if current and (
                len(current) >= self.batch_segments
                or current_characters + characters > self.batch_characters
                or estimate_batch_tokens(candidate) > self.batch_token_target
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
        scheduler: TranslationRateLimitScheduler | None = None,
    ) -> dict[int, str]:
        active_scheduler = scheduler or TranslationRateLimitScheduler()
        try:
            return translate_batch(
                items,
                os.environ["GROQ_API_KEY"],
                retry=retry,
                scheduler=active_scheduler,
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
            combined = self._translate_aligned(
                items[:midpoint],
                retry=retry,
                scheduler=active_scheduler,
            )
            combined.update(
                self._translate_aligned(
                    items[midpoint:],
                    retry=retry,
                    scheduler=active_scheduler,
                )
            )
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
                self.futures.pop(job_id, None)

    def _reconcile_future(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            future = self.futures.get(job_id)
            if (
                job is None
                or future is None
                or job.get("status") not in {"queued", "processing"}
                or not future.done()
            ):
                return
            try:
                failure = future.exception()
            except CancelledError as exc:
                failure = exc
            if failure is not None or job.get("status") in {"queued", "processing"}:
                job.update(
                    status="failed",
                    stage="failed",
                    public_error_code="INTERNAL_ERROR",
                    waiting=False,
                    wait_reason=None,
                    wait_until=None,
                    retry_after_seconds=0.0,
                    updated_at=time.time(),
                    finished_at=time.time(),
                )
                if self.active_job_id == job_id:
                    self.active_job_id = None

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
