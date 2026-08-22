"""Groq chat client for aligned English-to-Japanese segment translation."""

from __future__ import annotations

import json
import os

import httpx

from render_bridge_app.errors import BridgeError
from render_bridge_app.translation_rate_limit import (
    RateLimitSnapshot,
    TranslationRateLimitScheduler,
    completion_token_limit,
    estimate_request_token_ceiling,
)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TRANSLATION_MODEL = "openai/gpt-oss-120b"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_HTTP_RETRIES = 4


def translation_model() -> str:
    return os.getenv("GROQ_TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL)


def configured_max_completion_tokens() -> int:
    return max(
        512,
        min(
            8192,
            int(
                os.getenv(
                    "GROQ_TRANSLATION_MAX_COMPLETION_TOKENS",
                    str(DEFAULT_MAX_COMPLETION_TOKENS),
                )
            ),
        ),
    )


def max_completion_tokens(items: list[dict[str, object]] | None = None) -> int:
    configured = configured_max_completion_tokens()
    if items is None:
        return configured
    return completion_token_limit(items, configured)


def http_retries() -> int:
    return max(
        0,
        min(
            6,
            int(os.getenv("GROQ_TRANSLATION_HTTP_RETRIES", str(DEFAULT_HTTP_RETRIES))),
        ),
    )


def rate_limit_delay(response: httpx.Response, attempt: int) -> float:
    snapshot = RateLimitSnapshot.from_headers(getattr(response, "headers", {}))
    if snapshot.retry_after_seconds is not None:
        return max(1.0, min(60.0, snapshot.retry_after_seconds))
    if snapshot.reset_tokens_seconds is not None:
        return max(1.0, min(60.0, snapshot.reset_tokens_seconds))
    return min(30.0, float(2 ** (attempt + 1)))


def translate_batch(
    items: list[dict[str, object]],
    api_key: str,
    *,
    retry: bool = False,
    scheduler: TranslationRateLimitScheduler | None = None,
) -> dict[int, str]:
    """Translate an indexed batch and reject any alignment drift."""
    active_scheduler = scheduler or TranslationRateLimitScheduler()
    active_scheduler.record_batch(items)
    completion_limit = max_completion_tokens(items)
    request_token_ceiling = estimate_request_token_ceiling(
        items,
        completion_limit,
    )
    expected_ids = [int(item["id"]) for item in items]
    schema = {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "translation": {"type": "string"},
                    },
                    "required": ["id", "translation"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }
    retry_instruction = (
        " This is a single retry: correct every preservation mismatch exactly."
        if retry
        else ""
    )
    request_payload = {
        "model": translation_model(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate each English transcript segment into natural Japanese. "
                    "Return exactly one item for every input id in the same alignment. "
                    "Preserve all numbers, URLs, and the tickers BTC, ETH, SOL, XRP, "
                    "USDT, and USDC without changing their meaning or identity. "
                    "Do not add commentary."
                    + retry_instruction
                ),
            },
            {
                "role": "user",
                "content": json.dumps(items, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "temperature": 0,
        "reasoning_effort": "low",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "segment_translations",
                "strict": True,
                "schema": schema,
            },
        },
        "max_completion_tokens": completion_limit,
        "store": False,
    }
    try:
        request_timeout = active_scheduler.http_timeout_seconds()
        with httpx.Client(
            timeout=httpx.Timeout(request_timeout, connect=min(30, request_timeout)),
        ) as client:
            for attempt in range(http_retries() + 1):
                active_scheduler.before_request(request_token_ceiling)
                active_scheduler.request_started()
                response = client.post(
                    GROQ_CHAT_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=request_payload,
                )
                active_scheduler.observe_response(
                    response,
                    estimated_tokens=request_token_ceiling,
                )
                if response.status_code != 429 or attempt >= http_retries():
                    break
                if active_scheduler.daily_request_limit_reached():
                    raise BridgeError(
                        "GROQ_TRANSLATION_DAILY_LIMIT",
                        "Groq daily request limit was reached",
                    )
                active_scheduler.wait_for_retry(attempt)
    except httpx.HTTPError as exc:
        raise BridgeError(
            "GROQ_TRANSLATION_NETWORK_ERROR",
            "The Groq translation request failed",
        ) from exc
    if response.status_code in {401, 403}:
        raise BridgeError("GROQ_TRANSLATION_AUTH", "Groq rejected the API key")
    if response.status_code == 429:
        active_scheduler.mark_final_429_failure()
        if active_scheduler.daily_request_limit_reached():
            raise BridgeError(
                "GROQ_TRANSLATION_DAILY_LIMIT",
                "Groq daily request limit was reached",
            )
        raise BridgeError(
            "GROQ_TRANSLATION_RATE_LIMIT",
            "Groq translation rate limit was reached",
        )
    if response.status_code >= 400:
        raise BridgeError(
            "GROQ_TRANSLATION_ERROR",
            f"Groq returned HTTP {response.status_code}",
        )
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise BridgeError(
            "GROQ_TRANSLATION_RESPONSE_TOO_LARGE",
            "Groq translation response was too large",
        )
    try:
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(content)
        translations = payload["translations"]
        mapped = {
            int(item["id"]): str(item["translation"]).strip()
            for item in translations
        }
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeError(
            "GROQ_TRANSLATION_INVALID_RESPONSE",
            "Groq returned invalid translation JSON",
        ) from exc
    if sorted(mapped) != sorted(expected_ids) or len(mapped) != len(items):
        raise BridgeError(
            "GROQ_TRANSLATION_ALIGNMENT",
            "Groq translation alignment did not match the request",
        )
    if any(not mapped[item_id] for item_id in expected_ids):
        raise BridgeError(
            "GROQ_TRANSLATION_MISSING",
            "Groq returned an empty translation",
        )
    return mapped
