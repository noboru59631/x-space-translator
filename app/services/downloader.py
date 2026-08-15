"""Download audio and public metadata from X Space URLs."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from app.services.errors import DownloadError, InvalidSourceError

LOGGER = logging.getLogger(__name__)
ALLOWED_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
ALLOWED_PATHS = ("/i/spaces/", "/i/broadcasts/")


def normalize_x_url(url: str) -> str:
    """Validate and normalize an X Spaces/broadcast URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_HOSTS:
        raise InvalidSourceError("X SpacesのURLを入力してください。")
    if not any(parsed.path.startswith(prefix) for prefix in ALLOWED_PATHS):
        raise InvalidSourceError(
            "対応しているURLは /i/spaces/ または /i/broadcasts/ です。"
        )
    host = "x.com"
    return urlunparse(("https", host, parsed.path.rstrip("/"), "", "", ""))


def download_space(
    url: str, destination: Path, cookie_file: str = ""
) -> tuple[Path, dict[str, object]]:
    """Use yt-dlp to download the best available audio stream."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError(
            "yt-dlpがインストールされていません。setup.batを実行してください。"
        ) from exc

    normalized = normalize_x_url(url)
    destination.mkdir(parents=True, exist_ok=True)
    output_template = str(destination / "source.%(ext)s")
    options: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    if cookie_file:
        cookie_path = Path(cookie_file).expanduser().resolve()
        if cookie_path.is_file():
            options["cookiefile"] = str(cookie_path)
        else:
            LOGGER.warning("Configured cookie file was not found")
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(normalized, download=True)
            requested = info.get("requested_downloads") or []
            candidate = (
                Path(requested[0]["filepath"])
                if requested
                else Path(downloader.prepare_filename(info))
            )
    except Exception as exc:
        LOGGER.exception("X Space download failed")
        raise DownloadError(
            "X Spaceから音声を取得できませんでした。X側のアクセス制限、Cookieの必要性、URL、配信音声の公開状態を確認するか、音声ファイルをアップロードしてください。"
        ) from exc
    if not candidate.is_file():
        matches = list(destination.glob("source.*"))
        if not matches:
            raise DownloadError("ダウンロードした音声ファイルを確認できませんでした。")
        candidate = matches[0]
    metadata = {
        "title": info.get("title") or "取得不可",
        "host": info.get("uploader") or info.get("channel") or "取得不可",
        "upload_date": info.get("upload_date") or "取得不可",
        "webpage_url": normalized,
        "duration": info.get("duration") or 0,
    }
    return candidate, metadata
