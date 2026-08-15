"""Background job orchestration and cleanup."""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import Settings
from app.database import Store
from app.services import diarizer
from app.services.audio import convert_to_wav
from app.services.cache import file_cache_key, url_cache_key
from app.services.downloader import download_space
from app.services.errors import AppError
from app.services.transcriber import Transcriber
from app.services.translator import Translator

LOGGER = logging.getLogger(__name__)


class JobManager:
    """Run heavy local ML jobs outside FastAPI's request loop."""

    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audio-job"
        )
        self.transcriber = Transcriber()
        self.translator = Translator()

    def submit_transcription(self, job_id: str) -> None:
        self.executor.submit(self._process_transcription, job_id)

    def submit_translation(self, job_id: str) -> None:
        self.executor.submit(self._process_translation, job_id)

    def _cancelled(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        return not job or bool(job["cancel_requested"])

    def _set_progress(self, job_id: str, stage: str, progress: int) -> None:
        if self._cancelled(job_id):
            raise InterruptedError
        self.store.update_job(job_id, stage=stage, progress=progress)

    def _process_transcription(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if not job:
            return
        work_dir = self.settings.temp_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            source_path: Path
            metadata: dict[str, object] = {}
            if job["source_type"] == "url":
                self._set_progress(job_id, "downloading", 8)
                source_path, metadata = download_space(
                    job["source_url"], work_dir, self.settings.x_cookie_file
                )
                cache_key = url_cache_key(
                    job["source_url"], job["mode"], bool(job["diarize"])
                )
            else:
                self._set_progress(job_id, "validating", 8)
                source_path = Path(job["source_path"])
                cache_key = file_cache_key(
                    source_path, job["mode"], bool(job["diarize"])
                )

            cached_id = self.store.find_cached(cache_key)
            if cached_id:
                self.store.update_job(
                    job_id,
                    status="completed",
                    stage="cached",
                    progress=100,
                    completed_at=self._now(),
                    transcript_id=cached_id,
                )
                return

            self._set_progress(job_id, "converting", 20)
            wav_path = convert_to_wav(source_path, work_dir / "audio.wav")
            self._set_progress(job_id, "transcribing", 35)
            result = self.transcriber.transcribe(
                wav_path,
                job["mode"],
                self.settings.whisper_device,
                self.settings.whisper_model,
                self.settings.whisper_compute_type,
                lambda value: self._set_progress(job_id, "transcribing", value),
            )
            if bool(job["diarize"]):
                self._set_progress(job_id, "diarizing", 90)
                result["segments"] = diarizer.diarize(
                    wav_path, result["segments"], self.settings.hf_token
                )
            result.update(
                {
                    "source_url": job["source_url"],
                    "title": metadata.get(
                        "title",
                        Path(job["source_path"]).stem if job["source_path"] else "",
                    ),
                    "metadata": metadata,
                }
            )
            transcript_id = self.store.save_transcript(cache_key, result)
            self.store.update_job(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                completed_at=self._now(),
                transcript_id=transcript_id,
            )
        except InterruptedError:
            self.store.update_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                completed_at=self._now(),
                error="処理をキャンセルしました。",
            )
        except AppError as exc:
            LOGGER.exception("Expected processing error for job %s", job_id)
            self.store.update_job(
                job_id,
                status="failed",
                stage="failed",
                completed_at=self._now(),
                error=str(exc),
            )
        except Exception:
            LOGGER.exception("Unexpected processing error for job %s", job_id)
            self.store.update_job(
                job_id,
                status="failed",
                stage="failed",
                completed_at=self._now(),
                error="予期しないエラーが発生しました。ログで詳細を確認してください。",
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _process_translation(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if not job or not job.get("transcript_id"):
            return
        try:
            result = self.store.get_transcript(job["transcript_id"])
            if not result:
                raise RuntimeError("Transcript missing")
            self._set_progress(job_id, "translating", 92)
            translations = self.translator.translate(
                [segment["original"] for segment in result["segments"]]
            )
            self.store.update_translations(job["transcript_id"], translations)
            self.store.update_job(
                job_id, status="completed", stage="completed", progress=100, error=None
            )
        except AppError as exc:
            self.store.update_job(
                job_id,
                status="completed",
                stage="translation_failed",
                progress=100,
                error=str(exc),
            )
        except Exception:
            LOGGER.exception("Translation failed for job %s", job_id)
            self.store.update_job(
                job_id,
                status="completed",
                stage="translation_failed",
                progress=100,
                error="翻訳に失敗しました。英語の文字起こしは利用できます。",
            )

    @staticmethod
    def _now() -> str:
        from app.database.store import utc_now

        return utc_now()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
