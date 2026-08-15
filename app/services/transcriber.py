"""Faster Whisper transcription with automatic CPU/GPU selection."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Callable

from app.services.errors import DependencyError, ProcessingError

LOGGER = logging.getLogger(__name__)
MODELS = {"light": "base", "standard": "small", "accurate": "large-v3"}
MAX_CHUNK_SECONDS = 20 * 60


def split_wav(audio_path: Path) -> tuple[list[tuple[Path, float]], float]:
    """Split long PCM WAV files without loading the complete audio into memory."""
    with wave.open(str(audio_path), "rb") as source:
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        total_duration = total_frames / frame_rate
        frames_per_chunk = frame_rate * MAX_CHUNK_SECONDS
        if total_frames <= frames_per_chunk:
            return [(audio_path, 0.0)], total_duration

        chunk_dir = audio_path.parent / "whisper_chunks"
        chunk_dir.mkdir(exist_ok=True)
        parameters = source.getparams()
        chunks: list[tuple[Path, float]] = []
        frame_offset = 0
        index = 0
        while frame_offset < total_frames:
            frame_count = min(frames_per_chunk, total_frames - frame_offset)
            data = source.readframes(frame_count)
            if not data:
                break
            chunk_path = chunk_dir / f"chunk_{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as destination:
                destination.setparams(parameters)
                destination.writeframes(data)
            chunks.append((chunk_path, frame_offset / frame_rate))
            frame_offset += frame_count
            index += 1
    return chunks, total_duration


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
        self._key: tuple[str, str, str, int] | None = None

    def transcribe(
        self,
        audio_path: Path,
        mode: str,
        device_preference: str,
        model_override: str,
        compute_override: str,
        progress: Callable[[int], None] | None = None,
        cpu_threads: int = 0,
    ) -> dict[str, object]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyError(
                "faster-whisperがインストールされていません。setup.batを実行してください。"
            ) from exc
        device, compute = detect_device(device_preference)
        model_name = (
            model_override or "base"
            if mode == "cloud"
            else MODELS.get(mode, model_override or "small")
        )
        if device_preference != "auto" and compute_override != "auto":
            compute = compute_override
        key = (model_name, device, compute, cpu_threads)
        try:
            if self._model is None or self._key != key:
                self._model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute,
                    cpu_threads=cpu_threads,
                )
                self._key = key
            segments = []
            chunks, duration = split_wav(audio_path)
            detected_language = ""
            language_probability = None
            for chunk_path, offset in chunks:
                segments_iter, info = self._model.transcribe(
                    str(chunk_path),
                    beam_size=1 if mode in {"light", "cloud"} else 5,
                    vad_filter=True,
                    condition_on_previous_text=True,
                )
                if not detected_language:
                    detected_language = getattr(info, "language", "")
                    language_probability = getattr(info, "language_probability", None)
                for item in segments_iter:
                    text = item.text.strip()
                    if text:
                        segments.append(
                            {
                                "speaker": "Speaker A",
                                "start": offset + float(item.start),
                                "end": offset + float(item.end),
                                "original": text,
                                "translation_ja": "",
                            }
                        )
                    if progress and duration:
                        elapsed = offset + float(item.end)
                        progress(min(89, 35 + int((elapsed / duration) * 50)))
            return {
                "detected_language": detected_language,
                "language_probability": language_probability,
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
            if "allocate memory" in str(exc).lower() or "out of memory" in str(exc).lower():
                raise ProcessingError(
                    "処理に必要なメモリが不足しました。軽量モードを選択してください。"
                ) from exc
            LOGGER.exception("Transcription failed")
            raise ProcessingError(
                "文字起こしに失敗しました。ログで詳細を確認してください。"
            ) from exc
