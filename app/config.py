"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

BANKR_VIEWER_URL = (
    "https://bankr.bot/u/0x7b9af3d72ad97aa15db0e0cc6c1b747904653645/"
    "apps/space-youtube-transcriber"
)


class Settings(BaseSettings):
    """Runtime configuration with safe local-only defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8765
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    hf_token: str = ""
    x_cookie_file: str = ""
    bankr_viewer_url: HttpUrl = HttpUrl(BANKR_VIEWER_URL)
    max_upload_mb: int = Field(default=2048, ge=1)
    temp_dir: Path = Path("./temp")
    data_dir: Path = Path("./data")
    cors_origins: str = "http://127.0.0.1:8765,http://localhost:8765"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "x_space_translator.db"

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def prepare_directories(self) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_directories()
    return settings
