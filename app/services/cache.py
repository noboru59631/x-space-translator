"""Stable cache-key helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.services.downloader import normalize_x_url


def url_cache_key(url: str, mode: str, diarize: bool) -> str:
    value = f"url:{normalize_x_url(url)}:{mode}:{int(diarize)}"
    return hashlib.sha256(value.encode()).hexdigest()


def file_cache_key(path: Path, mode: str, diarize: bool) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(f":{mode}:{int(diarize)}".encode())
    return digest.hexdigest()
