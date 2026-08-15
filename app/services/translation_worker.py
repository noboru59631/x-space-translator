"""Isolated translation worker invoked by :mod:`app.services.translator`."""

from __future__ import annotations

import json
import os
import sys

MODEL_NAME = "facebook/m2m100_418M"


def main() -> None:
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    payload = json.load(sys.stdin)
    texts = payload["texts"]
    batch_size = min(int(payload.get("batch_size", 4)), 4)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
        model = M2M100ForConditionalGeneration.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
    except OSError:
        tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
        model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "en"
    translated: list[str] = []
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[offset : offset + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("ja"),
            max_new_tokens=128,
            num_beams=4,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        translated.extend(
            text.strip()
            for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        )
    json.dump(translated, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
