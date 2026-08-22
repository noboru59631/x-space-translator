from __future__ import annotations

import httpx
import pytest

from render_bridge_app.errors import BridgeError
from render_bridge_app.translation_jobs import TranslationJobManager
from render_bridge_app.translation_rate_limit import (
    RateLimitSnapshot,
    TranslationRateLimitScheduler,
    completion_token_limit,
    estimate_batch_tokens,
    estimate_request_token_ceiling,
    parse_duration_seconds,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def response(
    status_code: int,
    *,
    headers: dict[str, str] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers=headers,
        json={
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        },
        request=httpx.Request("POST", "https://api.groq.com/test"),
    )


def test_groq_duration_and_header_values_are_parsed_without_content() -> None:
    assert parse_duration_seconds("7.66s") == pytest.approx(7.66)
    assert parse_duration_seconds("2m59.56s") == pytest.approx(179.56)
    snapshot = RateLimitSnapshot.from_headers(
        {
            "retry-after": "2",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "1234",
            "x-ratelimit-reset-tokens": "7.66s",
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-reset-requests": "23h59m",
        }
    )
    assert snapshot.safe_values() == {
        "retry-after": 2.0,
        "x-ratelimit-limit-tokens": 8000,
        "x-ratelimit-remaining-tokens": 1234,
        "x-ratelimit-reset-tokens": pytest.approx(7.66),
        "x-ratelimit-limit-requests": 1000,
        "x-ratelimit-remaining-requests": 999,
        "x-ratelimit-reset-requests": pytest.approx(86340.0),
    }


def test_scheduler_waits_for_token_reset_when_next_batch_will_not_fit() -> None:
    clock = FakeClock()
    scheduler = TranslationRateLimitScheduler(
        sleeper=clock.sleep,
        clock=clock,
        fallback_interval_seconds=20,
    )
    scheduler.request_started()
    scheduler.observe_response(
        response(
            200,
            headers={
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "1000",
                "x-ratelimit-reset-tokens": "7.66s",
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "999",
            },
            prompt_tokens=400,
            completion_tokens=600,
        ),
        estimated_tokens=1200,
    )
    scheduler.before_request(estimated_tokens=1800)
    assert clock.waits == [pytest.approx(8.66)]
    assert scheduler.metrics()["header_wait_seconds"] == pytest.approx(8.66)


def test_scheduler_uses_safe_fixed_interval_without_headers() -> None:
    clock = FakeClock()
    scheduler = TranslationRateLimitScheduler(
        sleeper=clock.sleep,
        clock=clock,
        fallback_interval_seconds=20,
    )
    scheduler.request_started()
    scheduler.observe_response(
        response(200, prompt_tokens=300, completion_tokens=400),
        estimated_tokens=1000,
    )
    scheduler.before_request(estimated_tokens=1000)
    assert clock.waits == [20]
    assert scheduler.metrics()["fallback_wait_seconds"] == 20


def test_scheduler_local_window_stays_below_conservative_tpm_budget() -> None:
    clock = FakeClock()
    scheduler = TranslationRateLimitScheduler(
        token_limit=8000,
        token_budget=6000,
        sleeper=clock.sleep,
        clock=clock,
        fallback_interval_seconds=0,
    )
    for _ in range(2):
        scheduler.request_started()
        scheduler.observe_response(
            response(200, prompt_tokens=1500, completion_tokens=1500),
            estimated_tokens=3000,
        )
    scheduler.before_request(estimated_tokens=1000)
    assert clock.waits == [60]
    assert scheduler.metrics()["local_tpm_wait_seconds"] == 60


def test_429_retry_order_and_daily_request_detection() -> None:
    retry_clock = FakeClock()
    retry_scheduler = TranslationRateLimitScheduler(
        sleeper=retry_clock.sleep,
        clock=retry_clock,
    )
    retry_scheduler.observe_response(
        response(
            429,
            headers={
                "retry-after": "3",
                "x-ratelimit-reset-tokens": "9s",
                "x-ratelimit-remaining-requests": "5",
            },
        ),
        estimated_tokens=1000,
    )
    retry_scheduler.wait_for_retry(0)
    assert retry_clock.waits == [4]
    assert retry_scheduler.metrics()["rate_limit_429_count"] == 1

    reset_clock = FakeClock()
    reset_scheduler = TranslationRateLimitScheduler(
        sleeper=reset_clock.sleep,
        clock=reset_clock,
    )
    reset_scheduler.observe_response(
        response(
            429,
            headers={"x-ratelimit-reset-tokens": "9s"},
        ),
        estimated_tokens=1000,
    )
    reset_scheduler.wait_for_retry(0)
    assert reset_clock.waits == [10]

    daily = TranslationRateLimitScheduler()
    daily.observe_response(
        response(
            429,
            headers={
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "0",
            },
        ),
        estimated_tokens=1000,
    )
    assert daily.daily_request_limit_reached() is True


def test_long_rate_limit_headers_are_capped_and_rechecked() -> None:
    clock = FakeClock()
    events: list[tuple[str, dict[str, object]]] = []
    scheduler = TranslationRateLimitScheduler(
        sleeper=clock.sleep,
        clock=clock,
        wall_clock=clock,
        telemetry=lambda event, values: events.append((event, values)),
    )
    scheduler.observe_response(
        response(
            200,
            headers={
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "23m",
            },
            prompt_tokens=500,
            completion_tokens=750,
        ),
        estimated_tokens=1250,
    )
    scheduler.before_request(estimated_tokens=1250)
    assert clock.waits == [60]
    started = [values for event, values in events if event == "wait_started"]
    assert started[0]["wait_reason"] == "token_reset"
    assert started[0]["retry_after_seconds"] == 60

    for _ in range(2):
        scheduler.observe_response(
            response(429, headers={"retry-after": "1200"}),
            estimated_tokens=1250,
        )
        scheduler.wait_for_retry(0)
    assert clock.waits == [60, 60, 60]
    assert max(clock.waits) <= 60
    assert scheduler.metrics()["retry_count"] == 2
    assert scheduler.metrics()["longest_single_wait_seconds"] == 60


def test_scheduler_deadline_stops_repeated_waits() -> None:
    clock = FakeClock()
    scheduler = TranslationRateLimitScheduler(
        sleeper=clock.sleep,
        clock=clock,
        wall_clock=clock,
        deadline=60,
    )
    scheduler.observe_response(
        response(429, headers={"retry-after": "1200"}),
        estimated_tokens=1250,
    )
    with pytest.raises(BridgeError, match="processing deadline") as exc_info:
        scheduler.wait_for_retry(0)
    assert exc_info.value.code == "TRANSLATION_TIMEOUT"
    assert clock.waits == [60]


def test_free_plan_batches_and_estimate_leave_tpm_headroom() -> None:
    manager = TranslationJobManager(
        ttl_seconds=1800,
        batch_segments=10,
        batch_characters=2000,
    )
    items = [{"id": index, "text": "x" * 100} for index in range(289)]
    batches = manager._batches(items)
    assert len(batches) == 97
    assert max(len(batch) for batch in batches) == 3
    assert max(sum(len(str(item["text"])) for item in batch) for batch in batches) == 300
    assert max(estimate_batch_tokens(batch) for batch in batches) == 1206
    completion_limits = [completion_token_limit(batch, 4096) for batch in batches]
    ceilings = [
        estimate_request_token_ceiling(batch, completion_limit)
        for batch, completion_limit in zip(
            batches,
            completion_limits,
            strict=True,
        )
    ]
    assert max(completion_limits) == 768
    assert max(ceilings) == 1316
    assert max(ceilings) * 4 < 6000


def test_actual_usage_replaces_estimate_and_caps_rolling_window() -> None:
    clock = FakeClock()
    scheduler = TranslationRateLimitScheduler(
        token_budget=6000,
        sleeper=clock.sleep,
        clock=clock,
        fallback_interval_seconds=0,
    )
    for _ in range(4):
        scheduler.request_started()
        scheduler.observe_response(
            response(200, prompt_tokens=500, completion_tokens=800),
            estimated_tokens=1500,
        )
    assert scheduler.metrics()["max_rolling_60_second_tokens"] == 5200
    scheduler.before_request(estimated_tokens=1300)
    assert clock.waits == [60]


def test_normal_two_batch_fixture_waits_about_twenty_seconds() -> None:
    clock = FakeClock()
    scheduler = TranslationRateLimitScheduler(
        token_budget=6000,
        sleeper=clock.sleep,
        clock=clock,
        wall_clock=clock,
        fallback_interval_seconds=20,
    )
    for _ in range(2):
        scheduler.before_request(estimated_tokens=1250)
        scheduler.request_started()
        scheduler.observe_response(
            response(200, prompt_tokens=500, completion_tokens=750),
            estimated_tokens=1250,
        )
    assert clock.waits == [20]
    assert scheduler.metrics()["total_wait_seconds"] == 20
    assert scheduler.metrics()["max_rolling_60_second_tokens"] == 2500
