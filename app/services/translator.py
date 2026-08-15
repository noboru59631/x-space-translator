"""Local English-to-Japanese translation."""

from __future__ import annotations

import logging

from app.services.errors import DependencyError, ProcessingError

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "Helsinki-NLP/opus-mt-en-jap"


class Translator:
    def __init__(self) -> None:
        self._pipeline = None

    @staticmethod
    def available() -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def translate(self, texts: list[str], batch_size: int = 8) -> list[str]:
        if not self.available():
            raise DependencyError(
                "翻訳機能がインストールされていません。requirements.txtを再インストールしてください。"
            )
        try:
            if self._pipeline is None:
                from transformers import pipeline

                self._pipeline = pipeline("translation", model=MODEL_NAME, device=-1)
            translated: list[str] = []
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset : offset + batch_size]
                outputs = self._pipeline(batch, max_length=512)
                translated.extend(item["translation_text"].strip() for item in outputs)
            return translated
        except Exception as exc:
            LOGGER.exception("Translation failed")
            raise ProcessingError(
                "日本語翻訳に失敗しました。初回モデルダウンロード時はインターネット接続も確認してください。"
            ) from exc
