from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_home_and_health(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "X Space Translator" in home.text
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
