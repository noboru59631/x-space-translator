"""HTTP API routes."""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.models.schemas import (
    RenameSpeakersRequest,
    TranslationRequest,
    UrlTranscriptionRequest,
)
from app.services import diarizer
from app.services.audio import find_ffmpeg, validate_upload
from app.services.errors import AppError
from app.services.exports import render
from app.services.transcriber import detect_device
from app.services.translator import Translator

router = APIRouter(prefix="/api")


def context(request: Request):
    return request.app.state.settings, request.app.state.store, request.app.state.jobs


def public_job(job: dict) -> dict:
    return {
        key: job.get(key)
        for key in (
            "id",
            "status",
            "stage",
            "progress",
            "error",
            "created_at",
            "completed_at",
        )
    } | {"job_id": job["id"]}


@router.get("/health")
def health(request: Request) -> dict:
    settings, _, _ = context(request)
    device, _ = detect_device(settings.whisper_device)
    whisper = importlib.util.find_spec("faster_whisper") is not None
    return {
        "status": "ok",
        "ffmpeg": bool(find_ffmpeg()),
        "whisper": whisper,
        "gpu": device == "cuda",
        "diarization_available": diarizer.is_available(settings.hf_token),
        "translation_available": Translator.available(),
    }


@router.post("/transcribe/url", status_code=202)
def transcribe_url(payload: UrlTranscriptionRequest, request: Request) -> dict:
    _, store, jobs = context(request)
    job_id = uuid.uuid4().hex
    store.create_job(
        job_id,
        "url",
        source_url=payload.url,
        mode=payload.mode,
        diarize=payload.diarize,
    )
    jobs.submit_transcription(job_id)
    return {"job_id": job_id, "status": "processing"}


@router.post("/transcribe/file", status_code=202)
def transcribe_file(
    request: Request,
    file: Annotated[UploadFile, File()],
    mode: Annotated[Literal["light", "standard", "accurate"], Form()] = "standard",
    diarize: Annotated[bool, Form()] = False,
) -> dict:
    settings, store, jobs = context(request)
    try:
        extension = validate_upload(file.filename or "", file.content_type)
    except AppError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex
    work_dir = settings.temp_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=False)
    destination = work_dir / f"upload{extension}"
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        413,
                        f"ファイルサイズは{settings.max_upload_mb}MB以下にしてください。",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(400, "空のファイルは処理できません。")
        store.create_job(
            job_id, "file", source_path=str(destination), mode=mode, diarize=diarize
        )
        jobs.submit_transcription(job_id)
    except Exception:
        if not store.get_job(job_id):
            shutil.rmtree(work_dir, ignore_errors=True)
        raise
    return {"job_id": job_id, "status": "processing"}


@router.post("/translate", status_code=202)
def translate(payload: TranslationRequest, request: Request) -> dict:
    _, store, jobs = context(request)
    job = store.get_job(payload.job_id)
    if not job or job["status"] != "completed" or not job["transcript_id"]:
        raise HTTPException(409, "先に文字起こしを完了してください。")
    store.update_job(
        payload.job_id,
        status="processing",
        stage="translation_queued",
        progress=90,
        error=None,
    )
    jobs.submit_translation(payload.job_id)
    return {"job_id": payload.job_id, "status": "processing"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    _, store, _ = context(request)
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "ジョブが見つかりません。")
    return public_job(job)


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str, request: Request) -> dict:
    _, store, _ = context(request)
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "ジョブが見つかりません。")
    if not job["transcript_id"]:
        raise HTTPException(409, "処理はまだ完了していません。")
    result = store.get_transcript(job["transcript_id"])
    if not result:
        raise HTTPException(404, "文字起こし結果が見つかりません。")
    return result


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    _, store, _ = context(request)
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "ジョブが見つかりません。")
    if job["status"] == "processing":
        store.update_job(job_id, cancel_requested=1, stage="cancelling")
    return {
        "job_id": job_id,
        "status": "cancelling" if job["status"] == "processing" else job["status"],
    }


@router.put("/jobs/{job_id}/speakers")
def rename_speakers(
    job_id: str, payload: RenameSpeakersRequest, request: Request
) -> dict:
    _, store, _ = context(request)
    job = store.get_job(job_id)
    if not job or not job["transcript_id"]:
        raise HTTPException(404, "文字起こし結果が見つかりません。")
    store.rename_speakers(job["transcript_id"], payload.names)
    return store.get_transcript(job["transcript_id"])


@router.get("/jobs/{job_id}/export/{file_format}")
def export_result(
    job_id: str,
    file_format: Literal["txt", "srt", "vtt", "json"],
    request: Request,
    display: Literal["en", "ja", "both"] = "both",
) -> Response:
    _, store, _ = context(request)
    job = store.get_job(job_id)
    result = (
        store.get_transcript(job["transcript_id"])
        if job and job["transcript_id"]
        else None
    )
    if not result:
        raise HTTPException(404, "文字起こし結果が見つかりません。")
    content, media_type = render(result, file_format, display)
    filename = f"transcript-{job_id[:8]}.{file_format}"
    return Response(
        content=content,
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
