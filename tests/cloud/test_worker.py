from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.errors import DownloadError
from cloud.worker.app import main as worker


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(worker, "WORK_ROOT", tmp_path)
    worker.manager.jobs.clear()
    return TestClient(worker.app)


def test_health_and_url_validation(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "base"
    assert health.json()["compute_type"] == "int8"

    rejected = client.post("/transcribe", json={"url": "https://youtube.com/watch?v=x"})
    assert rejected.status_code == 422


def test_url_job_completion_and_cleanup(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(url: str, destination: Path) -> tuple[Path, dict[str, object]]:
        source = destination / "source.wav"
        source.write_bytes(b"audio")
        return source, {}

    def fake_transcribe(job_id: str, source: Path, job_dir: Path) -> None:
        worker.manager._update(
            job_id,
            status="completed",
            stage="completed",
            result={
                "status": "completed",
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
        "/transcribe", json={"url": "https://x.com/i/spaces/1qxvvvQBRXQxB/peek"}
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] == "completed" and job["audio_deleted"]:
            break
        import time

        time.sleep(0.01)
    assert job["detected_language"] == "en"
    assert job["segments"][0]["original"] == "hello"
    assert job["audio_deleted"] is True
    assert not (tmp_path / job_id).exists()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Sign in to confirm and provide cookies", "authentication_required"),
        ("HTTP Error 403: Forbidden", "datacenter_ip_block"),
        ("Unsupported URL: extractor failed", "yt_dlp_extractor"),
        ("This Space has expired", "expired_space"),
        ("Connection timed out", "network"),
        ("Unknown failure", "code_error"),
    ],
)
def test_download_failure_classification(message: str, expected: str) -> None:
    assert worker.classify_download_failure(DownloadError(message)) == expected
