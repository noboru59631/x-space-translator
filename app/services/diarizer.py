"""Optional pyannote speaker diarization."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def is_available(token: str) -> bool:
    if not token:
        return False
    try:
        import pyannote.audio  # noqa: F401

        return True
    except ImportError:
        return False


def _label_map(labels: list[str]) -> dict[str, str]:
    unique = sorted(set(labels))
    return {
        label: f"Speaker {chr(65 + index)}" if index < 26 else f"Speaker {index + 1}"
        for index, label in enumerate(unique)
    }


def diarize(
    audio_path: Path, segments: list[dict[str, object]], token: str
) -> list[dict[str, object]]:
    """Assign the overlapping diarization speaker to each Whisper segment.

    Failure is intentionally soft: the transcript remains usable as Speaker A.
    """
    if not is_available(token):
        return segments
    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        output = pipeline(str(audio_path))
        turns = [
            (float(turn.start), float(turn.end), label)
            for turn, _, label in output.itertracks(yield_label=True)
        ]
        labels = _label_map([turn[2] for turn in turns])
        for segment in segments:
            start, end = float(segment["start"]), float(segment["end"])
            best = max(
                turns,
                key=lambda turn: max(0.0, min(end, turn[1]) - max(start, turn[0])),
                default=None,
            )
            if best and min(end, best[1]) > max(start, best[0]):
                segment["speaker"] = labels[best[2]]
    except Exception:
        LOGGER.exception("Diarization failed; keeping single-speaker labels")
    return segments
