"""Disk-based M4A remuxing and ffprobe validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from render_bridge_app.errors import BridgeError, MediaValidationError

VALID_M4A_FORMATS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    size_bytes: int
    codec: str
    container: str


def dependencies_available() -> tuple[bool, bool]:
    return bool(shutil.which("ffmpeg")), bool(shutil.which("ffprobe"))


def remux_to_m4a(source: Path, destination: Path) -> Path:
    """Copy the first audio stream into an M4A container without re-encoding."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BridgeError("FFMPEG_MISSING", "FFmpeg is not available")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BridgeError(
            "REMUX_FAILED",
            "The audio could not be copied into a valid M4A container",
        ) from exc
    return destination


def probe_m4a(path: Path) -> MediaInfo:
    """Require a non-empty M4A container with one valid AAC audio stream."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise MediaValidationError("INVALID_MEDIA", "The M4A file is empty or missing")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise BridgeError("FFPROBE_MISSING", "ffprobe is not available")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = json.loads(completed.stdout)
    except (
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise MediaValidationError(
            "INVALID_MEDIA",
            "ffprobe could not validate the M4A file",
        ) from exc

    format_data = payload.get("format") or {}
    streams = payload.get("streams") or []
    format_name = str(format_data.get("format_name") or "")
    containers = {item.strip() for item in format_name.split(",") if item.strip()}
    if not containers.intersection(VALID_M4A_FORMATS):
        raise MediaValidationError(
            "INVALID_CONTAINER",
            "The remuxed file is not a valid M4A/MP4 container",
        )
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    if not audio_streams or audio_streams[0].get("codec_name") != "aac":
        raise MediaValidationError(
            "INVALID_AUDIO_CODEC",
            "The M4A file does not contain an AAC audio stream",
        )
    raw_duration = format_data.get("duration") or audio_streams[0].get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise MediaValidationError(
            "INVALID_DURATION",
            "The M4A duration is missing",
        ) from exc
    if duration <= 0:
        raise MediaValidationError(
            "INVALID_DURATION",
            "The M4A duration must be greater than zero",
        )
    return MediaInfo(
        duration=duration,
        size_bytes=path.stat().st_size,
        codec="aac",
        container=format_name,
    )
