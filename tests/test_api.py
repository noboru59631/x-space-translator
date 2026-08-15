from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_home_and_health(tmp_path, monkeypatch):
    bankr_url = (
        "https://bankr.bot/u/0x7b9af3d72ad97aa15db0e0cc6c1b747904653645/"
        "apps/space-youtube-transcriber"
    )
    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("BANKR_VIEWER_URL", bankr_url)
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "X Space Translator" in home.text
        assert "Bankrで結果を見る" in home.text
        assert f'href="{bankr_url}"' in home.text
        assert 'target="_blank"' in home.text
        assert 'rel="noopener noreferrer"' in home.text
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"


def test_rejects_non_x_url(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    with TestClient(app) as client:
        response = client.post(
            "/api/transcribe/url",
            json={
                "url": "https://example.com/audio",
                "mode": "light",
                "diarize": False,
            },
        )
        assert response.status_code == 422


def test_bankr_viewer_is_optional(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("BANKR_VIEWER_URL", "https://127.0.0.1:9/unavailable")

    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert 'href="https://127.0.0.1:9/unavailable"' in home.text
        assert client.get("/api/health").status_code == 200
