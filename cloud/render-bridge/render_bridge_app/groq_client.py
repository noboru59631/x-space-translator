"""Streaming multipart client for Groq speech-to-text."""

from __future__ import annotations

from pathlib import Path

import httpx

from render_bridge_app.errors import BridgeError

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024


def transcribe_m4a(path: Path, api_key: str) -> dict[str, object]:
    """Send an M4A file from disk and return normalized segment timestamps."""
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": GROQ_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "temperature": "0",
    }
    timeout = httpx.Timeout(900, connect=30, write=900, read=900)
    try:
        with path.open("rb") as audio:
            files = {"file": (path.name, audio, "audio/mp4")}
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    GROQ_TRANSCRIPTION_URL,
                    headers=headers,
                    data=data,
                    files=files,
                )
    except (httpx.HTTPError, OSError) as exc:
        raise BridgeError(
            "GROQ_NETWORK_ERROR",
            "The Groq transcription request failed",
        ) from exc
    if response.status_code in {401, 403}:
        raise BridgeError("GROQ_AUTH", "Groq rejected the API key")
    if response.status_code == 413:
        raise BridgeError("GROQ_FILE_TOO_LARGE", "Groq rejected the audio file size")
    if response.status_code == 429:
        raise BridgeError("GROQ_RATE_LIMIT", "Groq rate limit was reached")
    if response.status_code >= 400:
        raise BridgeError(
            "GROQ_ERROR",
            f"Groq returned HTTP {response.status_code}",
        )
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise BridgeError("GROQ_RESPONSE_TOO_LARGE", "Groq response was too large")
    try:
        payload = response.json()
    except ValueError as exc:
        raise BridgeError("GROQ_INVALID_RESPONSE", "Groq returned invalid JSON") from exc

    normalized_segments = []
    for segment in payload.get("segments") or []:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeError(
                "GROQ_INVALID_RESPONSE",
                "Groq returned an invalid segment",
            ) from exc
        normalized_segments.append(
            {
                "speaker": "Speaker",
                "start": round(start, 3),
                "end": round(end, 3),
                "original": text,
            }
        )
    try:
        duration = float(payload.get("duration") or 0)
    except (TypeError, ValueError) as exc:
        raise BridgeError(
            "GROQ_INVALID_RESPONSE",
            "Groq returned an invalid duration",
        ) from exc
    return {
        "detected_language": str(payload.get("language") or ""),
        "duration": round(duration, 3),
        "segments": normalized_segments,
    }
