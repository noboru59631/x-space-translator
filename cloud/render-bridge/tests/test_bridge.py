from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.errors import DownloadError
from render_bridge_app import groq_client, media
from render_bridge_app.errors import BridgeError, MediaValidationError
from render_bridge_app.main import app
from render_bridge_app import main as bridge

BRIDGE_KEY = "test-bridge-key"
AUTH = {"Authorization": f"Bearer {BRIDGE_KEY}"}
SPACE_URL = "https://x.com/i/spaces/1qxvvvQBRXQxB/peek"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    deadline = time.monotonic() + 2
    while bridge.manager.is_busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    monkeypatch.setenv("BRIDGE_API_KEY", BRIDGE_KEY)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(bridge, "WORK_ROOT", tmp_path)
    with bridge.manager.lock:
        bridge.manager.jobs.clear()
        bridge.manager.active_job_id = None
    return TestClient(app)


def test_health_is_public_and_minimal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "dependencies_available", lambda: (True, True))
    monkeypatch.setattr(
        bridge.importlib.util,
        "find_spec",
        lambda name: object() if name == "yt_dlp" else None,
    )
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ffmpeg": True,
        "yt_dlp": True,
    }


def test_authentication_and_missing_configuration(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"url": SPACE_URL}
    missing = client.post("/jobs", json=payload)
    wrong = client.post(
        "/jobs",
        json=payload,
        headers={"Authorization": "Bearer wrong"},
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    monkeypatch.delenv("BRIDGE_API_KEY")
    assert client.post("/jobs", json=payload, headers=AUTH).status_code == 503
    monkeypatch.setenv("BRIDGE_API_KEY", BRIDGE_KEY)
    monkeypatch.delenv("GROQ_API_KEY")
    assert client.post("/jobs", json=payload, headers=AUTH).status_code == 503


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/watch?v=x",
        "http://localhost/i/spaces/example",
        "http://127.0.0.1/i/spaces/example",
        "file:///etc/passwd",
        "ftp://x.com/i/spaces/example",
        "https://example.com/i/spaces/example",
    ],
)
def test_url_validation_rejects_non_x_and_ssrf(
    client: TestClient, url: str
) -> None:
    response = client.post("/jobs", json={"url": url}, headers=AUTH)
    assert response.status_code == 422


def test_twitter_url_is_normalized(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_submit(url: str) -> dict[str, object]:
        captured.append(url)
        return {"job_id": "test", "status": "queued"}

    monkeypatch.setattr(bridge.manager, "submit", fake_submit)
    response = client.post(
        "/transcribe",
        json={"url": "https://twitter.com/i/broadcasts/example/?secret=query"},
        headers=AUTH,
    )
    assert response.status_code == 202
    assert captured == ["https://x.com/i/broadcasts/example"]


def test_json_body_limit(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        content=b"x" * (bridge.MAX_JSON_BODY_BYTES + 1),
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_busy_control(client: TestClient) -> None:
    with bridge.manager.lock:
        bridge.manager.active_job_id = "active"
    response = client.post("/jobs", json={"url": SPACE_URL}, headers=AUTH)
    assert response.status_code == 429


def test_job_result_and_cleanup(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(
        url: str, destination: Path
    ) -> tuple[Path, dict[str, object]]:
        source = destination / "source.webm"
        source.write_bytes(b"source")
        return source, {"title": "Test Space"}

    def fake_remux(source: Path, destination: Path) -> Path:
        destination.write_bytes(b"m4a")
        return destination

    monkeypatch.setattr(bridge, "download_space", fake_download)
    monkeypatch.setattr(bridge, "remux_to_m4a", fake_remux)
    monkeypatch.setattr(
        bridge,
        "probe_m4a",
        lambda path: media.MediaInfo(11.0, 3, "aac", "mov,mp4,m4a"),
    )
    monkeypatch.setattr(
        bridge,
        "transcribe_m4a",
        lambda path, key: {
            "detected_language": "en",
            "duration": 11.0,
            "segments": [
                {
                    "speaker": "Speaker",
                    "start": 0.0,
                    "end": 1.0,
                    "original": "hello",
                }
            ],
        },
    )

    response = client.post("/jobs", json={"url": SPACE_URL}, headers=AUTH)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert job["status"] == "completed"
    assert job["cleanup"] is True
    assert job["media_codec"] == "aac"
    assert "segments" not in job
    result = client.get(f"/jobs/{job_id}/result", headers=AUTH).json()
    assert set(result) == {
        "title",
        "source_url",
        "detected_language",
        "duration",
        "segments",
    }
    assert result["segments"][0]["original"] == "hello"
    assert not (tmp_path / job_id).exists()


def test_oversized_media_never_calls_groq(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(
        url: str, destination: Path
    ) -> tuple[Path, dict[str, object]]:
        source = destination / "source.m4a"
        source.write_bytes(b"source")
        return source, {}

    def fake_remux(source: Path, destination: Path) -> Path:
        destination.write_bytes(b"m4a")
        return destination

    called = False

    def forbidden_groq(path: Path, key: str) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(bridge, "download_space", fake_download)
    monkeypatch.setattr(bridge, "remux_to_m4a", fake_remux)
    monkeypatch.setattr(
        bridge,
        "probe_m4a",
        lambda path: media.MediaInfo(
            100.0,
            bridge.groq_upload_limit_bytes() + 1,
            "aac",
            "mov,mp4,m4a",
        ),
    )
    monkeypatch.setattr(bridge, "transcribe_m4a", forbidden_groq)

    job_id = client.post(
        "/jobs",
        json={"url": SPACE_URL},
        headers=AUTH,
    ).json()["job_id"]
    for _ in range(200):
        job = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if job["status"] == "failed":
            break
        time.sleep(0.01)
    assert job["error_code"] == "GROQ_FILE_TOO_LARGE"
    assert job["cleanup"] is True
    assert called is False


def test_processing_result_does_not_expose_transcript(client: TestClient) -> None:
    with bridge.manager.lock:
        bridge.manager.jobs["processing"] = {
            "job_id": "processing",
            "status": "processing",
            "stage": "transcribing",
            "progress": 65,
            "created_at": time.time(),
        }
    response = client.get("/jobs/processing/result", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert "segments" not in response.json()


def test_probe_requires_aac_m4a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "audio.m4a"
    target.write_bytes(b"media")
    probe = {
        "format": {"format_name": "mov,mp4,m4a", "duration": "2.5"},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(probe),
        ),
    )
    info = media.probe_m4a(target)
    assert info.duration == 2.5
    assert info.codec == "aac"

    probe["streams"][0]["codec_name"] = "opus"
    with pytest.raises(MediaValidationError) as error:
        media.probe_m4a(target)
    assert error.value.code == "INVALID_AUDIO_CODEC"


def test_remux_uses_stream_copy_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.webm"
    target = tmp_path / "output.m4a"
    source.write_bytes(b"media")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(media.subprocess, "run", fake_run)
    assert media.remux_to_m4a(source, target) == target
    assert captured["command"][captured["command"].index("-c:a") + 1] == "copy"
    assert "shell" not in captured


def test_groq_response_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "audio.m4a"
    target.write_bytes(b"media")

    class FakeResponse:
        status_code = 200
        content = b"{}"

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "language": "en",
                "duration": 5.5,
                "segments": [{"start": 1, "end": 2.5, "text": " hello "}],
            }

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @staticmethod
        def post(*args: object, **kwargs: object) -> FakeResponse:
            assert kwargs["data"]["model"] == "whisper-large-v3-turbo"
            assert kwargs["data"]["response_format"] == "verbose_json"
            assert kwargs["data"]["timestamp_granularities[]"] == "segment"
            assert hasattr(kwargs["files"]["file"][1], "read")
            return FakeResponse()

    monkeypatch.setattr(groq_client.httpx, "Client", FakeClient)
    result = groq_client.transcribe_m4a(target, "secret")
    assert result == {
        "detected_language": "en",
        "duration": 5.5,
        "segments": [
            {
                "speaker": "Speaker",
                "start": 1.0,
                "end": 2.5,
                "original": "hello",
            }
        ],
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Sign in and provide cookies", "AUTH_REQUIRED"),
        ("HTTP 403 Forbidden", "BLOCKED"),
        ("Unsupported URL extractor", "EXTRACTOR_ERROR"),
        ("Connection timed out", "NETWORK_ERROR"),
        ("Unknown", "DOWNLOAD_ERROR"),
    ],
)
def test_download_failure_classification(message: str, expected: str) -> None:
    assert bridge.classify_download_failure(DownloadError(message)) == expected


def test_bridge_error_does_not_include_secrets() -> None:
    error = BridgeError("GROQ_ERROR", "Groq returned HTTP 500")
    assert "secret" not in str(error).lower()
