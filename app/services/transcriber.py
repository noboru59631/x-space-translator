"""Faster Whisper transcription with automatic CPU/GPU selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from app.services.errors import DependencyError, ProcessingError

LOGGER = logging.getLogger(__name__)
MODELS = {"light": "base", "standard": "small", "accurate": "large-v3"}


def detect_device(preference: str = "auto") -> tuple[str, str]:
    """Return a conservative faster-whisper device and compute type."""
    if preference in {"cpu", "cuda"}:
        return preference, "int8" if preference == "cpu" else "float16"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        LOGGER.debug("CUDA detection failed", exc_info=True)
    return "cpu", "int8"


class Transcriber:
    """Lazy-load and reuse one Whisper model configuration at a time."""

    def __init__(self) -> None:
        self._model = None
        self._key: tuple[str, str, str] | None = None

    def transcribe(
        self,
        audio_path: Path,
        mode: str,
        device_preference: str,
        model_override: str,
        compute_override: str,
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, object]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyError(
                "faster-whisperがインストールされていません。setup.batを実行してください。"
            ) from exc
        device, compute = detect_device(device_preference)
        model_name = MODELS.get(mode, model_override or "small")
        if device_preference != "auto" and compute_override != "auto":
            compute = compute_override
        key = (model_name, device, compute)
        try:
            if self._model is None or self._key != key:
                self._model = WhisperModel(
                    model_name, device=device, compute_type=compute
                )
                self._key = key
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                beam_size=1 if mode == "light" else 5,
                vad_filter=True,
                condition_on_previous_text=True,
            )
            segments = []
            duration = float(getattr(info, "duration", 0) or 0)
            for item in segments_iter:
                text = item.text.strip()
                if text:
                    segments.append(
                        {
                            "speaker": "Speaker A",
                            "start": float(item.start),
                            "end": float(item.end),
                            "original": text,
                            "translation_ja": "",
                        }
                    )
                if progress and duration:
                    progress(min(89, 35 + int((float(item.end) / duration) * 50)))
            return {
                "detected_language": getattr(info, "language", ""),
                "language_probability": getattr(info, "language_probability", None),
                "duration": duration or (segments[-1]["end"] if segments else 0),
                "segments": segments,
                "device": device,
                "model": model_name,
            }
        except MemoryError as exc:
            raise ProcessingError(
                "処理に必要なメモリが不足しました。軽量モードを選択してください。"
            ) from exc
        except (DependencyError, ProcessingError):
            raise
        except Exception as exc:
            LOGGER.exception("Transcription failed")
            raise ProcessingError(
                "文字起こしに失敗しました。ログで詳細を確認してください。"
            ) from exc
