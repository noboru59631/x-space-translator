from __future__ import annotations

import json
import re
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
    bridge.public_rate_limiter.clear()
    with bridge.translation_manager.lock:
        bridge.translation_manager.jobs.clear()
        bridge.translation_manager.active_job_id = None
    bridge.public_translation_rate_limiter.clear()
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


def test_api_job_requires_authentication(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"url": SPACE_URL})
    assert response.status_code == 401
    assert client.get("/api/jobs/unknown").status_code == 401
    assert client.get("/api/jobs/unknown/result").status_code == 401


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


def test_api_job_rejects_invalid_url(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        json={"url": "http://127.0.0.1/i/spaces/example"},
        headers=AUTH,
    )
    assert response.status_code == 422
    assert response.json() == {"error_code": "INVALID_URL"}


def test_twitter_url_is_normalized(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_submit(
        url: str,
        *,
        visibility: str = "authenticated",
    ) -> dict[str, object]:
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
    public_response = client.post(
        "/public/jobs",
        content=b"x" * (bridge.MAX_JSON_BODY_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert public_response.status_code == 413


def test_busy_control(client: TestClient) -> None:
    with bridge.manager.lock:
        bridge.manager.active_job_id = "active"
    response = client.post("/jobs", json={"url": SPACE_URL}, headers=AUTH)
    assert response.status_code == 429
    api_response = client.post("/api/jobs", json={"url": SPACE_URL}, headers=AUTH)
    assert api_response.status_code == 429
    assert api_response.json() == {"detail": "BUSY"}


def test_api_job_returns_quickly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bridge.manager,
        "submit",
        lambda url, **kwargs: {"job_id": "a" * 32, "status": "queued"},
    )
    started = time.perf_counter()
    response = client.post("/api/jobs", json={"url": SPACE_URL}, headers=AUTH)
    elapsed = time.perf_counter() - started
    assert response.status_code == 202
    assert response.json() == {"job_id": "a" * 32, "status": "queued"}
    assert elapsed < 1.0


def test_job_id_is_random_uuid(client: TestClient) -> None:
    job = bridge.manager._reserve(SPACE_URL)
    try:
        assert re.fullmatch(r"[0-9a-f]{32}", str(job["job_id"]))
    finally:
        bridge.manager._abandon(str(job["job_id"]))


def test_public_job_needs_no_secret_and_returns_quickly(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_submit(
        url: str,
        *,
        visibility: str = "authenticated",
    ) -> dict[str, object]:
        captured["url"] = url
        captured["visibility"] = visibility
        return {"job_id": "b" * 32, "status": "queued"}

    monkeypatch.setattr(bridge.manager, "submit", fake_submit)
    started = time.perf_counter()
    response = client.post(
        "/public/jobs",
        json={"url": SPACE_URL},
        headers={"X-Forwarded-For": "198.51.100.10"},
    )
    assert time.perf_counter() - started < 1.0
    assert response.status_code == 202
    assert response.json() == {"job_id": "b" * 32, "status": "queued"}
    assert captured == {"url": SPACE_URL, "visibility": "public"}


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
def test_public_job_rejects_arbitrary_urls(client: TestClient, url: str) -> None:
    response = client.post("/public/jobs", json={"url": url})
    assert response.status_code == 422
    assert response.json() == {"error_code": "INVALID_URL"}


def test_public_rate_limit_is_per_ip(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = 0

    def fake_submit(url: str, **kwargs: object) -> dict[str, object]:
        nonlocal counter
        counter += 1
        return {"job_id": f"{counter:032x}", "status": "queued"}

    monkeypatch.setattr(bridge.manager, "submit", fake_submit)
    headers = {"X-Forwarded-For": "198.51.100.11"}
    assert client.post("/public/jobs", json={"url": SPACE_URL}, headers=headers).status_code == 202
    assert client.post("/public/jobs", json={"url": SPACE_URL}, headers=headers).status_code == 202
    limited = client.post("/public/jobs", json={"url": SPACE_URL}, headers=headers)
    assert limited.status_code == 429
    assert limited.json() == {"detail": "RATE_LIMITED"}
    assert int(limited.headers["retry-after"]) > 0
    other = client.post(
        "/public/jobs",
        json={"url": SPACE_URL},
        headers={"X-Forwarded-For": "198.51.100.12"},
    )
    assert other.status_code == 202


def test_public_client_ip_uses_render_first_forwarded_address(
    client: TestClient,
) -> None:
    request = client.build_request(
        "POST",
        "/public/jobs",
        headers={"X-Forwarded-For": "198.51.100.15, 10.0.0.2"},
    )
    assert bridge.public_client_ip(request) == "198.51.100.15"


def test_busy_public_submission_does_not_consume_rate_limit(
    client: TestClient,
) -> None:
    headers = {"X-Forwarded-For": "198.51.100.13"}
    with bridge.manager.lock:
        bridge.manager.active_job_id = "active"
    for _ in range(3):
        response = client.post(
            "/public/jobs",
            json={"url": SPACE_URL},
            headers=headers,
        )
        assert response.status_code == 429
        assert response.json() == {"detail": "BUSY"}


def test_public_routes_only_expose_public_jobs(client: TestClient) -> None:
    now = time.time()
    with bridge.manager.lock:
        bridge.manager.jobs["public"] = {
            "job_id": "public",
            "status": "completed",
            "stage": "completed",
            "visibility": "public",
            "created_at": now,
            "finished_at": now,
            "result": {"segments": []},
        }
        bridge.manager.jobs["private"] = {
            "job_id": "private",
            "status": "completed",
            "stage": "completed",
            "visibility": "authenticated",
            "created_at": now,
            "finished_at": now,
            "result": {"segments": []},
        }
    assert client.get("/public/jobs/public").status_code == 200
    assert client.get("/public/jobs/public/result").status_code == 200
    assert client.get("/public/jobs/private").status_code == 404
    assert client.get("/public/jobs/private/result").status_code == 404


def test_api_job_status_has_null_progress(client: TestClient) -> None:
    with bridge.manager.lock:
        bridge.manager.jobs["status"] = {
            "job_id": "status",
            "status": "processing",
            "stage": "validating_audio",
            "progress": None,
            "created_at": time.time(),
        }
    response = client.get("/api/jobs/status", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "status",
        "status": "processing",
        "stage": "validating_audio",
        "progress": None,
    }


def test_api_completed_result_is_wrapped(client: TestClient) -> None:
    transcript = {
        "title": "Test",
        "source_url": SPACE_URL,
        "detected_language": "en",
        "duration": 1.0,
        "segments": [],
    }
    with bridge.manager.lock:
        bridge.manager.jobs["done"] = {
            "job_id": "done",
            "status": "completed",
            "stage": "completed",
            "created_at": time.time(),
            "finished_at": time.time(),
            "result": transcript,
        }
    response = client.get("/api/jobs/done/result", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "done",
        "status": "completed",
        "result": transcript,
    }


def test_api_processing_result_is_minimal(client: TestClient) -> None:
    with bridge.manager.lock:
        bridge.manager.jobs["waiting"] = {
            "job_id": "waiting",
            "status": "processing",
            "stage": "transcribing",
            "created_at": time.time(),
        }
    response = client.get("/api/jobs/waiting/result", headers=AUTH)
    assert response.json() == {"job_id": "waiting", "status": "processing"}


def test_api_failed_job_exposes_only_public_error(client: TestClient) -> None:
    with bridge.manager.lock:
        bridge.manager.jobs["failed"] = {
            "job_id": "failed",
            "status": "failed",
            "stage": "failed",
            "created_at": time.time(),
            "finished_at": time.time(),
            "public_error_code": "GROQ_TRANSCRIPTION_FAILED",
            "error": "internal details must stay private",
        }
    response = client.get("/api/jobs/failed", headers=AUTH)
    assert response.json()["error_code"] == "GROQ_TRANSCRIPTION_FAILED"
    assert "error" not in response.json()


def test_api_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/jobs/missing", headers=AUTH).status_code == 404


def test_completed_job_expires_after_ttl(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "JOB_TTL_SECONDS", 600)
    with bridge.manager.lock:
        bridge.manager.jobs["expired"] = {
            "job_id": "expired",
            "status": "completed",
            "created_at": time.time() - 1200,
            "finished_at": time.time() - 601,
        }
    assert bridge.manager.get("expired") is None


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


def test_public_duration_limit_never_calls_groq_and_cleans_up(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(
        url: str,
        destination: Path,
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

    monkeypatch.setattr(bridge, "PUBLIC_MAX_AUDIO_SECONDS", 60)
    monkeypatch.setattr(bridge, "download_space", fake_download)
    monkeypatch.setattr(bridge, "remux_to_m4a", fake_remux)
    monkeypatch.setattr(
        bridge,
        "probe_m4a",
        lambda path: media.MediaInfo(61.0, 3, "aac", "mov,mp4,m4a"),
    )
    monkeypatch.setattr(bridge, "transcribe_m4a", forbidden_groq)

    response = client.post(
        "/public/jobs",
        json={"url": SPACE_URL},
        headers={"X-Forwarded-For": "198.51.100.14"},
    )
    job_id = response.json()["job_id"]
    for _ in range(200):
        job = client.get(f"/public/jobs/{job_id}").json()
        if job["status"] == "failed":
            break
        time.sleep(0.01)
    result = client.get(f"/public/jobs/{job_id}/result").json()
    assert job["error_code"] == "AUDIO_TOO_LONG"
    assert result["error_code"] == "AUDIO_TOO_LONG"
    assert called is False
    assert not (bridge.WORK_ROOT / job_id).exists()


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


@pytest.mark.parametrize(
    ("internal", "public"),
    [
        ("AUTH_REQUIRED", "X_DOWNLOAD_FAILED"),
        ("REMUX_FAILED", "AUDIO_INVALID"),
        ("GROQ_RATE_LIMIT", "GROQ_TRANSCRIPTION_FAILED"),
        ("CODE_ERROR", "INTERNAL_ERROR"),
    ],
)
def test_public_error_classification(internal: str, public: str) -> None:
    assert bridge.public_error_code(internal) == public
