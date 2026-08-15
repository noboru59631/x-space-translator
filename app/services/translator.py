"""Local English-to-Japanese translation."""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys

from app.services.errors import DependencyError, ProcessingError

LOGGER = logging.getLogger(__name__)


class Translator:
    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("transformers") is not None

    def translate(self, texts: list[str], batch_size: int = 4) -> list[str]:
        """Translate in a short-lived process so PyTorch memory is fully released."""
        if not self.available():
            raise DependencyError(
                "翻訳機能がインストールされていません。requirements.txtを再インストールしてください。"
            )
        try:
            process = subprocess.run(
                [sys.executable, "-m", "app.services.translation_worker"],
                input=json.dumps({"texts": texts, "batch_size": batch_size}),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            if process.returncode != 0:
                LOGGER.error("Translation worker failed: %s", process.stderr[-2000:])
                raise RuntimeError("Translation worker failed")
            translations = json.loads(process.stdout)
            if not isinstance(translations, list) or len(translations) != len(texts):
                raise RuntimeError("Translation worker returned an invalid result")
            return [str(text) for text in translations]
        except Exception as exc:
            LOGGER.exception("Translation failed")
            raise ProcessingError(
                "日本語翻訳に失敗しました。初回モデルダウンロード時はインターネット接続も確認してください。"
            ) from exc
