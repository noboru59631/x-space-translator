"""Isolated translation worker invoked by :mod:`app.services.translator`."""

from __future__ import annotations

import json
import sys

MODEL_NAME = "staka/fugumt-en-ja"


def main() -> None:
    from transformers import pipeline

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    payload = json.load(sys.stdin)
    texts = payload["texts"]
    batch_size = int(payload.get("batch_size", 8))
    translator = pipeline("translation", model=MODEL_NAME, device=-1)
    translated: list[str] = []
    for offset in range(0, len(texts), batch_size):
        outputs = translator(
            texts[offset : offset + batch_size],
            max_new_tokens=96,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        translated.extend(item["translation_text"].strip() for item in outputs)
    json.dump(translated, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
