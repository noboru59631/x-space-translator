"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class UrlTranscriptionRequest(BaseModel):
    url: str
    mode: Literal["light", "standard", "accurate"] = "standard"
    diarize: bool = False

    @field_validator("url")
    @classmethod
    def validate_x_url(cls, value: str) -> str:
        from app.services.downloader import normalize_x_url
        from app.services.errors import AppError

        try:
            normalize_x_url(value)
        except AppError as exc:
            raise ValueError(str(exc)) from exc
        return value


class TranslationRequest(BaseModel):
    job_id: str


class RenameSpeakersRequest(BaseModel):
    names: dict[str, str] = Field(default_factory=dict)

    @field_validator("names")
    @classmethod
    def validate_names(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for speaker, name in value.items():
            name = name.strip()
            if name:
                cleaned[speaker] = name[:80]
        return cleaned


class Segment(BaseModel):
    speaker: str = "Speaker A"
    start: float
    end: float
    original: str
    translation_ja: str = ""


class TranscriptResult(BaseModel):
    title: str = ""
    source_url: str = ""
    detected_language: str = ""
    language_probability: float | None = None
    duration: float = 0
    segments: list[Segment] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: str = "queued"
    progress: int = 0
    error: str | None = None
