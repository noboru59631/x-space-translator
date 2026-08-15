"""FFmpeg discovery, validation, and memory-friendly audio conversion."""

from __future__ import annotations

import mimetypes
import shutil
import subprocess
from pathlib import Path

from app.services.errors import DependencyError, InvalidSourceError, ProcessingError

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".webm"}
ALLOWED_MIME_PREFIXES = ("audio/", "video/")


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def validate_upload(filename: str, content_type: str | None) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidSourceError("対応形式は MP3 / WAV / M4A / MP4 / WEBM です。")
    guessed, _ = mimetypes.guess_type(filename)
    mime = content_type or guessed or ""
    if (
        mime
        and not mime.startswith(ALLOWED_MIME_PREFIXES)
        and mime != "application/octet-stream"
    ):
        raise InvalidSourceError("音声または動画ファイルを選択してください。")
    return extension


def convert_to_wav(source: Path, destination: Path) -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise DependencyError(
            "FFmpegが見つかりません。READMEのセットアップ手順を確認してください。"
        )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=None
        )
    except subprocess.CalledProcessError as exc:
        raise ProcessingError(
            "音声を変換できませんでした。ファイルが破損していないか確認してください。"
        ) from exc
    return destination
