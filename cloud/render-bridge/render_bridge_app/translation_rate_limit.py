"""Token-aware Groq Free Plan pacing without retaining request content."""

from __future__ import annotations

import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Mapping

import httpx

from render_bridge_app.errors import BridgeError

RATE_LIMIT_HEADER_NAMES = (
    "retry-after",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
)
DEFAULT_TOKEN_LIMIT = 8000
DEFAULT_TOKEN_BUDGET = 6000
DEFAULT_TOKEN_RESERVE = 2000
DEFAULT_FALLBACK_INTERVAL_SECONDS = 20.0
DEFAULT_BATCH_TOKEN_TARGET = 1250
DEFAULT_PROMPT_AND_SCHEMA_TOKENS = 400
DEFAULT_REASONING_TOKEN_ALLOWANCE = 256
DEFAULT_COMPLETION_TOKEN_FLOOR = 768
DEFAULT_COMPLETION_TOKEN_CEILING = 1536
TOKEN_WINDOW_SECONDS = 60.0
WINDOW_SAFETY_SECONDS = 1.0
MAX_SINGLE_RATE_LIMIT_WAIT_SECONDS = 60.0


def parse_int_header(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value.strip()))
    except ValueError:
        return None


def parse_duration_seconds(value: str | None) -> float | None:
    """Parse Groq reset durations such as ``7.66s`` or ``2m59.56s``."""
    if not value:
        return None
    stripped = value.strip().lower()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|d|h|m|s)", stripped)
    if not matches:
        return None
    multipliers = {
        "ms": 0.001,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }
    return sum(float(number) * multipliers[unit] for number, unit in matches)


def parse_retry_after_seconds(value: str | None) -> float | None:
    parsed = parse_duration_seconds(value)
    if parsed is not None:
        return parsed
    if not value:
        return None
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


@dataclass(frozen=True)
class RateLimitSnapshot:
    retry_after_seconds: float | None = None
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    reset_tokens_seconds: float | None = None
    limit_requests: int | None = None
    remaining_requests: int | None = None
    reset_requests_seconds: float | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> RateLimitSnapshot:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        return cls(
            retry_after_seconds=parse_retry_after_seconds(
                normalized.get("retry-after")
            ),
            limit_tokens=parse_int_header(
                normalized.get("x-ratelimit-limit-tokens")
            ),
            remaining_tokens=parse_int_header(
                normalized.get("x-ratelimit-remaining-tokens")
            ),
            reset_tokens_seconds=parse_duration_seconds(
                normalized.get("x-ratelimit-reset-tokens")
            ),
            limit_requests=parse_int_header(
                normalized.get("x-ratelimit-limit-requests")
            ),
            remaining_requests=parse_int_header(
                normalized.get("x-ratelimit-remaining-requests")
            ),
            reset_requests_seconds=parse_duration_seconds(
                normalized.get("x-ratelimit-reset-requests")
            ),
        )

    def safe_values(self) -> dict[str, int | float | None]:
        return {
            "retry-after": self.retry_after_seconds,
            "x-ratelimit-limit-tokens": self.limit_tokens,
            "x-ratelimit-remaining-tokens": self.remaining_tokens,
            "x-ratelimit-reset-tokens": self.reset_tokens_seconds,
            "x-ratelimit-limit-requests": self.limit_requests,
            "x-ratelimit-remaining-requests": self.remaining_requests,
            "x-ratelimit-reset-requests": self.reset_requests_seconds,
        }


def estimate_prompt_tokens(items: list[dict[str, object]]) -> int:
    """Estimate the system prompt, schema, instructions, and English input."""
    characters = sum(len(str(item["text"])) for item in items)
    json_overhead = len(items) * 16
    return (
        DEFAULT_PROMPT_AND_SCHEMA_TOKENS
        + math.ceil(characters / 3)
        + json_overhead
    )


def estimate_completion_tokens(items: list[dict[str, object]]) -> int:
    """Estimate Japanese JSON output plus low-effort reasoning tokens."""
    characters = sum(len(str(item["text"])) for item in items)
    json_overhead = len(items) * 24
    return (
        DEFAULT_REASONING_TOKEN_ALLOWANCE
        + math.ceil(characters * 1.1)
        + json_overhead
    )


def estimate_batch_tokens(items: list[dict[str, object]]) -> int:
    """Estimate all input and output tokens used by one translation request."""
    return estimate_prompt_tokens(items) + estimate_completion_tokens(items)


def completion_token_limit(items: list[dict[str, object]], configured: int) -> int:
    """Bound completion size to translation output rather than a global 4K cap."""
    dynamic = max(DEFAULT_COMPLETION_TOKEN_FLOOR, estimate_completion_tokens(items))
    return min(configured, DEFAULT_COMPLETION_TOKEN_CEILING, dynamic)


def estimate_request_token_ceiling(
    items: list[dict[str, object]],
    completion_limit: int,
) -> int:
    """Return a safe request ceiling using the actual completion token cap."""
    return estimate_prompt_tokens(items) + completion_limit


class TranslationRateLimitScheduler:
    """Pace translation requests using response headers and a local TPM window."""

    def __init__(
        self,
        *,
        token_limit: int | None = None,
        token_budget: int | None = None,
        token_reserve: int | None = None,
        fallback_interval_seconds: float | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        deadline: float | None = None,
        telemetry: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self.token_limit = token_limit or int(
            os.getenv("GROQ_TRANSLATION_TPM_LIMIT", str(DEFAULT_TOKEN_LIMIT))
        )
        configured_budget = token_budget or int(
            os.getenv("GROQ_TRANSLATION_TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET))
        )
        self.token_budget = min(self.token_limit, configured_budget)
        self.token_reserve = (
            token_reserve
            if token_reserve is not None
            else int(
                os.getenv(
                    "GROQ_TRANSLATION_TOKEN_RESERVE",
                    str(DEFAULT_TOKEN_RESERVE),
                )
            )
        )
        self.fallback_interval_seconds = (
            fallback_interval_seconds
            if fallback_interval_seconds is not None
            else float(
                os.getenv(
                    "GROQ_TRANSLATION_FALLBACK_INTERVAL_SECONDS",
                    str(DEFAULT_FALLBACK_INTERVAL_SECONDS),
                )
            )
        )
        self.sleep = sleeper or time.sleep
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.deadline = deadline
        self.telemetry = telemetry
        self.max_single_wait_seconds = max(
            1.0,
            min(
                MAX_SINGLE_RATE_LIMIT_WAIT_SECONDS,
                float(
                    os.getenv(
                        "TRANSLATION_MAX_SINGLE_WAIT_SECONDS",
                        str(MAX_SINGLE_RATE_LIMIT_WAIT_SECONDS),
                    )
                ),
            ),
        )
        self.snapshot = RateLimitSnapshot()
        self.request_tokens: deque[tuple[float, int]] = deque()
        self.last_request_at: float | None = None
        self.skip_next_provider_wait = False
        self.batch_count = 0
        self.batch_segments = 0
        self.batch_characters = 0
        self.requests_sent = 0
        self.successful_requests = 0
        self.rate_limit_429_count = 0
        self.retry_count = 0
        self.rate_limit_wait_count = 0
        self.final_429_failure = False
        self.total_wait_seconds = 0.0
        self.header_wait_seconds = 0.0
        self.fallback_wait_seconds = 0.0
        self.local_tpm_wait_seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.max_rolling_tokens = 0
        self.longest_single_wait_seconds = 0.0

    def record_batch(self, items: list[dict[str, object]]) -> None:
        self.batch_count += 1
        self.batch_segments += len(items)
        self.batch_characters += sum(len(str(item["text"])) for item in items)

    def before_request(self, estimated_tokens: int) -> None:
        self.ensure_time_remaining()
        now = self.clock()
        self._discard_expired_token_events(now)
        local_delay = self._local_window_delay(now, estimated_tokens)
        provider_delay = 0.0
        fallback_delay = 0.0
        used_header = False
        if self.skip_next_provider_wait:
            self.skip_next_provider_wait = False
        elif self.snapshot.remaining_tokens is not None:
            required = estimated_tokens + self.token_reserve
            if self.snapshot.remaining_tokens < required:
                provider_delay = (
                    self.snapshot.reset_tokens_seconds + WINDOW_SAFETY_SECONDS
                    if self.snapshot.reset_tokens_seconds is not None
                    else self.fallback_interval_seconds
                )
                used_header = self.snapshot.reset_tokens_seconds is not None
        elif self.last_request_at is not None:
            fallback_delay = max(
                0.0,
                self.last_request_at + self.fallback_interval_seconds - now,
            )

        delay = max(local_delay, provider_delay, fallback_delay)
        if delay <= 0:
            return
        if delay == local_delay:
            reason = "rolling_token_budget"
        elif used_header and delay == provider_delay:
            reason = "token_reset"
        else:
            reason = "pacing"
        self._wait(delay, reason=reason)
        if used_header and delay >= provider_delay:
            self.snapshot = replace(
                self.snapshot,
                remaining_tokens=self.snapshot.limit_tokens or self.token_limit,
                reset_tokens_seconds=None,
            )

    def request_started(self) -> None:
        self.ensure_time_remaining()
        self.requests_sent += 1
        self.last_request_at = self.clock()
        self._emit(
            "request_started",
            requests_sent=self.requests_sent,
            last_request_at=self.wall_clock(),
        )

    def observe_response(
        self,
        response: httpx.Response,
        *,
        estimated_tokens: int,
    ) -> None:
        headers = getattr(response, "headers", {})
        self.snapshot = RateLimitSnapshot.from_headers(headers)
        if response.status_code == 429:
            self.rate_limit_429_count += 1
            self._emit(
                "rate_limited",
                rate_limit_count=self.rate_limit_429_count,
                last_429_at=self.wall_clock(),
            )
            return
        prompt, completion, total = self._safe_usage(response)
        accounted_tokens = total or estimated_tokens
        self.successful_requests += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += accounted_tokens
        self.request_tokens.append((self.clock(), accounted_tokens))
        rolling_tokens = sum(tokens for _, tokens in self.request_tokens)
        self.max_rolling_tokens = max(self.max_rolling_tokens, rolling_tokens)
        self._emit(
            "request_succeeded",
            successful_requests=self.successful_requests,
            last_success_at=self.wall_clock(),
        )

    def mark_final_429_failure(self) -> None:
        self.final_429_failure = True

    def daily_request_limit_reached(self) -> bool:
        return (
            self.snapshot.remaining_requests is not None
            and self.snapshot.remaining_requests <= 0
        )

    def wait_for_retry(self, attempt: int) -> None:
        if self.snapshot.retry_after_seconds is not None:
            delay = self.snapshot.retry_after_seconds
            reason = "retry_after"
        elif self.snapshot.reset_tokens_seconds is not None:
            delay = self.snapshot.reset_tokens_seconds
            reason = "token_reset"
        else:
            delay = min(30.0, float(2 ** (attempt + 1)))
            reason = "exponential_backoff"
        self.retry_count += 1
        self._emit("retry", retry_count=self.retry_count)
        self._wait(max(1.0, delay + WINDOW_SAFETY_SECONDS), reason=reason)
        self.skip_next_provider_wait = True
        self.snapshot = replace(
            self.snapshot,
            remaining_tokens=None,
            reset_tokens_seconds=None,
            retry_after_seconds=None,
        )

    def safe_headers(self) -> dict[str, int | float | None]:
        return self.snapshot.safe_values()

    def metrics(self) -> dict[str, int | float | bool]:
        batch_count = self.batch_count or 1
        successful_requests = self.successful_requests or 1
        return {
            "batch_count": self.batch_count,
            "average_segments_per_batch": round(
                self.batch_segments / batch_count,
                2,
            ),
            "average_characters_per_batch": round(
                self.batch_characters / batch_count,
                2,
            ),
            "requests_sent": self.requests_sent,
            "successful_requests": self.successful_requests,
            "rate_limit_429_count": self.rate_limit_429_count,
            "retry_count": self.retry_count,
            "rate_limit_wait_count": self.rate_limit_wait_count,
            "final_429_failure": self.final_429_failure,
            "total_wait_seconds": round(self.total_wait_seconds, 3),
            "header_wait_seconds": round(self.header_wait_seconds, 3),
            "fallback_wait_seconds": round(self.fallback_wait_seconds, 3),
            "local_tpm_wait_seconds": round(self.local_tpm_wait_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "average_tokens_per_request": round(
                self.total_tokens / successful_requests,
                2,
            ),
            "max_rolling_60_second_tokens": self.max_rolling_tokens,
            "longest_single_wait_seconds": round(
                self.longest_single_wait_seconds,
                3,
            ),
        }

    def ensure_time_remaining(self) -> None:
        if self.deadline is not None and self.clock() >= self.deadline:
            raise BridgeError(
                "TRANSLATION_TIMEOUT",
                "Translation job exceeded its processing deadline",
            )

    def http_timeout_seconds(self, default: float = 180.0) -> float:
        if self.deadline is None:
            return default
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            self.ensure_time_remaining()
        return max(1.0, min(default, remaining))

    def _local_window_delay(self, now: float, estimated_tokens: int) -> float:
        used = sum(tokens for _, tokens in self.request_tokens)
        if used + estimated_tokens <= self.token_budget:
            return 0.0
        remaining_used = used
        for timestamp, tokens in self.request_tokens:
            remaining_used -= tokens
            if remaining_used + estimated_tokens <= self.token_budget:
                return max(
                    0.0,
                    timestamp
                    + TOKEN_WINDOW_SECONDS
                    + WINDOW_SAFETY_SECONDS
                    - now,
                )
        return TOKEN_WINDOW_SECONDS + WINDOW_SAFETY_SECONDS

    def _discard_expired_token_events(self, now: float) -> None:
        cutoff = now - TOKEN_WINDOW_SECONDS
        while self.request_tokens and self.request_tokens[0][0] <= cutoff:
            self.request_tokens.popleft()

    def _wait(self, delay: float, *, reason: str) -> None:
        if delay <= 0:
            return
        self.ensure_time_remaining()
        actual_delay = min(delay, self.max_single_wait_seconds)
        if self.deadline is not None:
            actual_delay = min(actual_delay, max(0.0, self.deadline - self.clock()))
        if actual_delay <= 0:
            self.ensure_time_remaining()
        self.total_wait_seconds += actual_delay
        self.rate_limit_wait_count += 1
        self.longest_single_wait_seconds = max(
            self.longest_single_wait_seconds,
            actual_delay,
        )
        if reason in {"retry_after", "token_reset"}:
            self.header_wait_seconds += actual_delay
        elif reason == "rolling_token_budget":
            self.local_tpm_wait_seconds += actual_delay
        else:
            self.fallback_wait_seconds += actual_delay
        self._emit(
            "wait_started",
            waiting=True,
            wait_reason=reason,
            wait_until=self.wall_clock() + actual_delay,
            retry_after_seconds=round(actual_delay, 3),
            rate_limit_wait_count=self.rate_limit_wait_count,
            longest_single_wait_seconds=round(
                self.longest_single_wait_seconds,
                3,
            ),
        )
        self.sleep(actual_delay)
        self._emit(
            "wait_finished",
            waiting=False,
            wait_reason=None,
            wait_until=None,
            retry_after_seconds=0.0,
        )
        self.ensure_time_remaining()

    def _emit(self, event: str, **values: object) -> None:
        if self.telemetry is not None:
            self.telemetry(event, values)

    @staticmethod
    def _safe_usage(response: httpx.Response) -> tuple[int, int, int]:
        try:
            usage = response.json().get("usage") or {}
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or prompt + completion)
            return prompt, completion, total
        except (AttributeError, TypeError, ValueError):
            return 0, 0, 0
