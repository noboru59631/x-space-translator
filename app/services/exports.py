"""TXT, SRT, VTT, and JSON rendering."""

from __future__ import annotations

import json


def timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def selected_text(segment: dict, display: str) -> str:
    original = segment.get("original", "")
    japanese = segment.get("translation_ja", "") or "（未翻訳）"
    if display == "ja":
        return japanese
    if display == "both":
        return f"{original}\n{japanese}"
    return original


def to_txt(result: dict, display: str) -> str:
    blocks = [
        f"{item['speaker']} [{timestamp(item['start'], '.')[:-4]}]\n\n{selected_text(item, display)}"
        for item in result["segments"]
    ]
    return "\n\n".join(blocks) + "\n"


def to_srt(result: dict, display: str) -> str:
    blocks = []
    for index, item in enumerate(result["segments"], 1):
        blocks.append(
            f"{index}\n{timestamp(item['start'])} --> {timestamp(item['end'])}\n{item['speaker']}: {selected_text(item, display)}"
        )
    return "\n\n".join(blocks) + "\n"


def to_vtt(result: dict, display: str) -> str:
    blocks = []
    for index, item in enumerate(result["segments"], 1):
        blocks.append(
            f"{index}\n{timestamp(item['start'], '.')} --> {timestamp(item['end'], '.')}\n"
            f"{item['speaker']}: {selected_text(item, display)}"
        )
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def render(result: dict, file_format: str, display: str) -> tuple[str, str]:
    if file_format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2), "application/json"
    if file_format == "srt":
        return to_srt(result, display), "application/x-subrip"
    if file_format == "vtt":
        return to_vtt(result, display), "text/vtt"
    return to_txt(result, display), "text/plain"
