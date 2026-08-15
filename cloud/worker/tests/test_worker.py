from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.errors import DownloadError
from cloud.worker.app import main as worker

API_KEY = "test-worker-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    deadline = time.monotonic() + 2
    while worker.manager.is_busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path)
    with worker.manager.lock:
        worker.manager.jobs.clear()
        worker.manager.active_job_id = None
    return TestClient(worker.app)


def test_health_is_public_and_reports_dependencies(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()) == {"status", "ffmpeg", "yt_dlp", "whisper"}
    assert all(isinstance(response.json()[name], bool) for name in ("ffmpeg", "yt_dlp", "whisper"))


def test_job_endpoints_require_valid_bearer_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = {"url": "https://x.com/i/spaces/1qxvvvQBRXQxB/peek"}
    missing = client.post("/jobs", json=url)
    wrong = client.post("/jobs", json=url, headers={"Authorization": "Bearer wrong"})
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    monkeypatch.delenv("WORKER_API_KEY")
    unavailable = client.post("/jobs", json=url, headers=AUTH)
    assert unavailable.status_code == 503


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=x",
        "http://127.0.0.1/i/spaces/example",
        "http://localhost/i/spaces/example",
        "file:///etc/passwd",
        "ftp://x.com/i/spaces/example",
        "https://example.com/i/spaces/example",
    ],
)
def test_url_allowlist_rejects_non_x_and_ssrf_targets(
    client: TestClient, url: str
) -> None:
    response = client.post("/jobs", json={"url": url}, headers=AUTH)
    assert response.status_code == 422


def test_json_request_body_limit(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        content=b"x" * (worker.MAX_JSON_BODY_BYTES + 1),
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_busy_worker_returns_429(client: TestClient) -> None:
    with worker.manager.lock:
        worker.manager.active_job_id = "already-active"
    response = client.post(
        "/jobs",
        json={"url": "https://x.com/i/spaces/1qxvvvQBRXQxB/peek"},
        headers=AUTH,
    )
    assert response.status_code == 429


def test_url_job_status_result_and_cleanup(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(url: str, destination: Path) -> tuple[Path, dict[str, object]]:
        source = destination / "source.wav"
        source.write_bytes(b"audio")
        return source, {"title": "Test Space"}

    def fake_transcribe(
        job_id: str,
        source: Path,
        job_dir: Path,
        *,
        title: str,
        source_url: str,
    ) -> None:
        worker.manager._update(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            result={
                "status": "completed",
                "title": title,
                "source_url": source_url,
                "detected_language": "en",
                "duration": 1.0,
                "segments": [
                    {"start": 0.0, "end": 1.0, "speaker": "Speaker", "original": "hello"}
                ],
            },
        )

    monkeypatch.setattr(worker, "download_space", fake_download)
    monkeypatch.setattr(worker.manager, "_transcribe", fake_transcribe)
    response = client.post(
        "/jobs",
        json={"url": "https://twitter.com/i/spaces/1qxvvvQBRXQxB/peek?ignored=yes"},
        headers=AUTH,
    )
    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "processing"}
    job_id = response.json()["job_id"]

    for _ in range(200):
        status_response = client.get(f"/jobs/{job_id}", headers=AUTH)
        job_status = status_response.json()
        if job_status["status"] == "completed" and job_status["audio_deleted"]:
            break
        time.sleep(0.01)

    assert "segments" not in job_status
    assert job_status["progress"] == 100
    result = client.get(f"/jobs/{job_id}/result", headers=AUTH).json()
    assert result["title"] == "Test Space"
    assert result["source_url"] == "https://x.com/i/spaces/1qxvvvQBRXQxB/peek"
    assert result["segments"][0]["original"] == "hello"
    assert job_status["audio_deleted"] is True
    assert not (tmp_path / job_id).exists()


def test_result_endpoint_reports_processing_without_transcript(client: TestClient) -> None:
    job = {
        "job_id": "processing-job",
        "status": "processing",
        "stage": "transcribing",
        "progress": 55,
        "error_code": None,
        "error": None,
        "elapsed_seconds": None,
        "peak_memory_mb": None,
        "audio_deleted": False,
        "created_at": time.time(),
        "result": None,
    }
    with worker.manager.lock:
        worker.manager.jobs["processing-job"] = job
    response = client.get("/jobs/processing-job/result", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["progress"] == 55
    assert "segments" not in response.json()


def test_invalid_upload_does_not_create_a_job(client: TestClient) -> None:
    response = client.post(
        "/jobs/file",
        files={"file": ("unsafe.txt", b"not audio", "text/plain")},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert worker.manager.jobs == {}
    assert worker.manager.is_busy() is False


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/jobs/missing", headers=AUTH).status_code == 404
    assert client.get("/jobs/missing/result", headers=AUTH).status_code == 404


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Sign in to confirm and provide cookies", "AUTH_REQUIRED"),
        ("HTTP Error 403: Forbidden", "BLOCKED"),
        ("Unsupported URL: extractor failed", "EXTRACTOR_ERROR"),
        ("This Space has expired", "EXPIRED_SPACE"),
        ("Connection timed out", "NETWORK_ERROR"),
        ("Unknown failure", "CODE_ERROR"),
    ],
)
def test_download_failure_classification(message: str, expected: str) -> None:
    assert worker.classify_download_failure(DownloadError(message)) == expected
