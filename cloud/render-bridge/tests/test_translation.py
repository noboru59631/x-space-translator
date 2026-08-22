from __future__ import annotations

import json
import re
import time
from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from render_bridge_app import main as bridge
from render_bridge_app import translation_client, translation_jobs
from render_bridge_app.main import app
from render_bridge_app.translation_rate_limit import TranslationRateLimitScheduler
from render_bridge_app.translation_validation import validate_preservation

SEGMENT = {
    "speaker": "Speaker",
    "start": 0,
    "end": 10,
    "original": "BTC reached 1 million dollars at https://example.com/a.",
}


def store_public_transcript(
    job_id: str,
    segments: list[dict[str, object]],
    *,
    status: str = "completed",
    finished_at: float | None = None,
) -> None:
    now = time.time()
    with bridge.manager.lock:
        bridge.manager.jobs[job_id] = {
            "job_id": job_id,
            "status": status,
            "stage": status,
            "visibility": "public",
            "created_at": now,
            "finished_at": finished_at if finished_at is not None else now,
            "result": {"segments": segments} if status == "completed" else None,
            "translation_job_id": None,
            "translation_jobs": {},
        }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    deadline = time.monotonic() + 2
    while bridge.translation_manager.is_busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    with bridge.manager.lock:
        bridge.manager.jobs.clear()
        bridge.manager.active_job_id = None
    with bridge.translation_manager.lock:
        bridge.translation_manager.jobs.clear()
        bridge.translation_manager.active_job_id = None
    bridge.public_rate_limiter.clear()
    bridge.public_translation_rate_limiter.clear()
    return TestClient(app)


def test_natural_number_formats_do_not_warn() -> None:
    check = validate_preservation(
        "BTC moved from 1 million to 2,024 and then ３.",
        "BTCは100万から2,024、そして三へ動きました。",
    )
    assert check.number_ok is True
    assert check.ticker_ok is True


def test_alphanumeric_and_word_numbers_match_japanese_digits() -> None:
    check = validate_preservation(
        "Layer 2 has five users in the first L2 group for 18 months.",
        "レイヤー2には、最初のL2グループで5人のユーザーが18か月います。",
    )
    assert check.number_ok is True


def test_month_and_repeated_source_number_are_natural() -> None:
    assert validate_preservation("Targeting August.", "8月を目標にします。").number_ok
    assert validate_preservation(
        "18 months passed in 18 months.",
        "18か月が経過しました。",
    ).number_ok


def test_added_or_missing_number_still_warns() -> None:
    assert not validate_preservation(
        "It takes one month.",
        "1か月は1年のように感じます。",
    ).number_ok
    assert not validate_preservation(
        "The first share is 20%.",
        "1位のシェアです。",
    ).number_ok


@pytest.mark.parametrize(
    ("translation", "field"),
    [
        ("BTCは200万ドルです。", "number_ok"),
        ("BTCです。", "url_ok"),
        ("ETHは100万ドルです。https://example.com/a", "ticker_ok"),
    ],
)
def test_preservation_mismatches_are_classified(
    translation: str,
    field: str,
) -> None:
    check = validate_preservation(SEGMENT["original"], translation)
    assert getattr(check, field) is False
    assert check.ok is False


def test_groq_translation_client_uses_strict_json_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        content = b"{}"

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"id": 7, "translation": "こんにちは"}
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @staticmethod
        def post(*args: object, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr(translation_client.httpx, "Client", FakeClient)
    result = translation_client.translate_batch(
        [{"id": 7, "text": "Hello"}],
        "secret-value",
    )
    assert result == {7: "こんにちは"}
    request_json = captured["json"]
    assert request_json["model"] == "openai/gpt-oss-120b"
    assert request_json["response_format"]["json_schema"]["strict"] is True
    assert captured["headers"]["Authorization"] == "Bearer secret-value"
    assert "secret-value" not in json.dumps(request_json)


def test_groq_rate_limit_uses_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.headers = {"retry-after": "1.5"}
            self.content = b"{}"

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"id": 1, "translation": "成功"}
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        calls = 0

        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @classmethod
        def post(cls, *args: object, **kwargs: object) -> FakeResponse:
            cls.calls += 1
            return FakeResponse(429 if cls.calls == 1 else 200)

    monkeypatch.setattr(translation_client.httpx, "Client", FakeClient)
    scheduler = TranslationRateLimitScheduler(
        sleeper=waits.append,
        clock=lambda: 0.0,
    )
    result = translation_client.translate_batch(
        [{"id": 1, "text": "Success"}],
        "secret",
        scheduler=scheduler,
    )
    assert result == {1: "成功"}
    assert waits == [2.5]


def test_translation_post_is_public_and_quick(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_submit(segments: list[dict[str, object]]) -> dict[str, object]:
        captured.extend(segments)
        return {"job_id": "a" * 32, "status": "queued"}

    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    started = time.perf_counter()
    response = client.post(
        "/public/translations",
        json={"segments": [SEGMENT]},
        headers={"X-Forwarded-For": "198.51.100.30"},
    )
    assert time.perf_counter() - started < 1
    assert response.status_code == 202
    assert response.json() == {"job_id": "a" * 32, "status": "queued"}
    assert captured[0]["original"] == SEGMENT["original"]


def test_completed_transcript_queues_job_id_translation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    context: dict[str, object] = {}

    def fake_submit(
        segments: list[dict[str, object]],
        *,
        result_context: dict[str, object] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        captured.extend(segments)
        context.update(result_context or {})
        return {"job_id": "b" * 32, "status": "queued"}

    store_public_transcript("transcript", [SEGMENT])
    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    response = client.post(
        "/public/jobs/transcript/translations",
        headers={"X-Forwarded-For": "198.51.100.40"},
    )
    assert response.status_code == 202
    assert response.json() == {"job_id": "b" * 32, "status": "queued"}
    assert captured == [{**SEGMENT, "index": 0}]
    assert context == {
        "start_index": 0,
        "count": 1,
        "total_segments": 1,
        "has_more": False,
        "next_index": None,
    }
    assert bridge.manager.jobs["transcript"]["translation_job_id"] == "b" * 32


def test_job_id_translation_defaults_to_first_20_of_289_segments(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = [{**SEGMENT, "start": float(index), "end": float(index + 1)} for index in range(289)]
    captured: list[dict[str, object]] = []
    context: dict[str, object] = {}

    def fake_submit(
        items: list[dict[str, object]],
        *,
        result_context: dict[str, object] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        captured.extend(items)
        context.update(result_context or {})
        return {"job_id": "c" * 32, "status": "queued"}

    store_public_transcript("full", segments)
    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    response = client.post(
        "/public/jobs/full/translations",
        json={},
        headers={"X-Forwarded-For": "198.51.100.41"},
    )
    assert response.status_code == 202
    assert len(captured) == 20
    assert captured[0]["original"] == SEGMENT["original"]
    assert captured[0]["index"] == 0
    assert captured[-1]["index"] == 19
    assert context == {
        "start_index": 0,
        "count": 20,
        "total_segments": 289,
        "has_more": True,
        "next_index": 20,
    }


@pytest.mark.parametrize("count", [1, 25])
def test_job_id_translation_accepts_range_count_limits(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    segments = [{**SEGMENT, "start": float(index)} for index in range(40)]
    captured: list[dict[str, object]] = []

    def fake_submit(
        items: list[dict[str, object]],
        *,
        result_context: dict[str, object] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        captured.extend(items)
        return {"job_id": f"{count:032x}", "status": "queued"}

    store_public_transcript(f"count-{count}", segments)
    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    response = client.post(
        f"/public/jobs/count-{count}/translations",
        json={"start_index": 0, "count": count},
    )
    assert response.status_code == 202
    assert len(captured) == count
    assert [item["index"] for item in captured] == list(range(count))


def test_job_id_translation_rejects_count_26_and_invalid_start(
    client: TestClient,
) -> None:
    store_public_transcript("invalid-range", [SEGMENT] * 30)
    too_many = client.post(
        "/public/jobs/invalid-range/translations",
        json={"start_index": 0, "count": 26},
    )
    negative = client.post(
        "/public/jobs/invalid-range/translations",
        json={"start_index": -1, "count": 20},
    )
    past_end = client.post(
        "/public/jobs/invalid-range/translations",
        json={"start_index": 30, "count": 20},
    )
    assert too_many.status_code == 422
    assert negative.status_code == 422
    assert past_end.status_code == 422
    assert past_end.json() == {"detail": "INVALID_START_INDEX"}


@pytest.mark.parametrize(
    ("job_id", "start_index", "expected_count", "has_more", "next_index"),
    [
        ("middle-range", 100, 20, True, 120),
        ("final-range", 280, 9, False, None),
    ],
)
def test_job_id_translation_range_metadata_and_original_indexes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    job_id: str,
    start_index: int,
    expected_count: int,
    has_more: bool,
    next_index: int | None,
) -> None:
    segments = [{**SEGMENT, "start": float(index)} for index in range(289)]
    captured: list[dict[str, object]] = []
    context: dict[str, object] = {}

    def fake_submit(
        items: list[dict[str, object]],
        *,
        result_context: dict[str, object] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        captured.extend(items)
        context.update(result_context or {})
        return {"job_id": "f" * 32, "status": "queued"}

    store_public_transcript(job_id, segments)
    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    response = client.post(
        f"/public/jobs/{job_id}/translations",
        json={"start_index": start_index, "count": 20},
    )
    assert response.status_code == 202
    assert len(captured) == expected_count
    assert captured[0]["index"] == start_index
    assert captured[-1]["index"] == start_index + expected_count - 1
    assert context == {
        "start_index": start_index,
        "count": expected_count,
        "total_segments": 289,
        "has_more": has_more,
        "next_index": next_index,
    }


def test_job_id_translation_safe_transcript_errors(client: TestClient) -> None:
    missing = client.post("/public/jobs/missing/translations")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "TRANSCRIPT_JOB_NOT_FOUND"}

    store_public_transcript("processing", [], status="processing")
    processing = client.post("/public/jobs/processing/translations")
    assert processing.status_code == 409
    assert processing.json() == {"detail": "TRANSCRIPT_NOT_READY"}

    store_public_transcript("empty", [])
    empty = client.post("/public/jobs/empty/translations")
    assert empty.status_code == 422
    assert empty.json() == {"detail": "TRANSCRIPT_EMPTY"}
    assert "secret" not in json.dumps([missing.json(), processing.json(), empty.json()])


@pytest.mark.parametrize("translation_status", ["queued", "processing", "completed"])
def test_job_id_translation_reuses_existing_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    translation_status: str,
) -> None:
    store_public_transcript("duplicate", [SEGMENT])
    translation_job_id = "d" * 32
    with bridge.manager.lock:
        bridge.manager.jobs["duplicate"]["translation_job_id"] = translation_job_id
    with bridge.translation_manager.lock:
        bridge.translation_manager.jobs[translation_job_id] = {
            "job_id": translation_job_id,
            "status": translation_status,
            "stage": translation_status,
            "created_at": time.time(),
            "finished_at": time.time() if translation_status == "completed" else None,
            "result": {"segments": []} if translation_status == "completed" else None,
        }

    def forbidden_submit(
        segments: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        raise AssertionError("duplicate request must not start another translation")

    monkeypatch.setattr(bridge.translation_manager, "submit", forbidden_submit)
    first = client.post("/public/jobs/duplicate/translations")
    second = client.post("/public/jobs/duplicate/translations")
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json() == {
        "job_id": translation_job_id,
        "status": translation_status,
    }
    assert second.json() == first.json()
    assert len(bridge.translation_manager.jobs) == 1
    assert bridge.public_translation_rate_limiter.events == {}


def test_failed_job_id_translation_is_not_duplicated(client: TestClient) -> None:
    store_public_transcript("failed-link", [SEGMENT])
    translation_job_id = "e" * 32
    with bridge.manager.lock:
        bridge.manager.jobs["failed-link"]["translation_job_id"] = translation_job_id
    with bridge.translation_manager.lock:
        bridge.translation_manager.jobs[translation_job_id] = {
            "job_id": translation_job_id,
            "status": "failed",
            "stage": "failed",
            "created_at": time.time(),
            "finished_at": time.time(),
            "result": None,
        }
    response = client.post("/public/jobs/failed-link/translations")
    assert response.status_code == 409
    assert response.json() == {"detail": "TRANSLATION_ALREADY_EXISTS"}


def test_completed_range_does_not_block_a_different_range(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_public_transcript("next-range", [SEGMENT] * 40)
    completed_job_id = "1" * 32
    with bridge.manager.lock:
        bridge.manager.jobs["next-range"]["translation_jobs"] = {
            "0:20": completed_job_id
        }
    with bridge.translation_manager.lock:
        bridge.translation_manager.jobs[completed_job_id] = {
            "job_id": completed_job_id,
            "status": "completed",
            "stage": "completed",
            "created_at": time.time(),
            "finished_at": time.time(),
            "result": {"segments": []},
        }

    captured: list[dict[str, object]] = []

    def fake_submit(
        segments: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        captured.extend(segments)
        return {"job_id": "2" * 32, "status": "queued"}

    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    response = client.post(
        "/public/jobs/next-range/translations",
        json={"start_index": 20, "count": 20},
        headers={"X-Forwarded-For": "198.51.100.51"},
    )
    assert response.status_code == 202
    assert len(captured) == 20
    assert captured[0]["index"] == 20
    assert captured[-1]["index"] == 39


def test_expired_transcript_cannot_start_job_id_translation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge, "JOB_TTL_SECONDS", 1)
    store_public_transcript(
        "expired-transcript",
        [SEGMENT],
        finished_at=time.time() - 2,
    )
    response = client.post("/public/jobs/expired-transcript/translations")
    assert response.status_code == 404
    assert response.json() == {"detail": "TRANSCRIPT_JOB_NOT_FOUND"}


def test_job_id_translation_uses_public_translation_rate_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = 0

    def fake_submit(
        segments: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal counter
        counter += 1
        return {"job_id": f"{counter:032x}", "status": "queued"}

    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    for job_id in ("rate-one", "rate-two", "rate-three"):
        store_public_transcript(job_id, [SEGMENT])
    headers = {"X-Forwarded-For": "198.51.100.42"}
    assert client.post("/public/jobs/rate-one/translations", headers=headers).status_code == 202
    assert client.post("/public/jobs/rate-two/translations", headers=headers).status_code == 202
    limited = client.post("/public/jobs/rate-three/translations", headers=headers)
    assert limited.status_code == 429
    assert limited.json() == {"detail": "RATE_LIMITED"}
    assert "retry-after" in limited.headers
    assert counter == 2


def test_translation_request_limits(client: TestClient) -> None:
    too_many = [SEGMENT] * (bridge.PUBLIC_TRANSLATION_MAX_SEGMENTS + 1)
    response = client.post(
        "/public/translations",
        json={"segments": too_many},
    )
    assert response.status_code == 422
    assert response.json() == {"error_code": "INVALID_TRANSLATION_REQUEST"}

    long_segments = [
        {**SEGMENT, "original": "x" * 10000}
        for _ in range(bridge.PUBLIC_TRANSLATION_MAX_CHARACTERS // 10000 + 1)
    ]
    response = client.post(
        "/public/translations",
        json={"segments": long_segments},
    )
    assert response.status_code == 422

    oversized = client.post(
        "/public/translations",
        content=b"x" * (bridge.MAX_TRANSLATION_JSON_BODY_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413


def test_translation_rate_limit_is_separate_and_per_ip(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = 0

    def fake_submit(segments: list[dict[str, object]]) -> dict[str, object]:
        nonlocal counter
        counter += 1
        return {"job_id": f"{counter:032x}", "status": "queued"}

    monkeypatch.setattr(bridge.translation_manager, "submit", fake_submit)
    headers = {"X-Forwarded-For": "198.51.100.31"}
    for _ in range(2):
        assert client.post(
            "/public/translations",
            json={"segments": [SEGMENT]},
            headers=headers,
        ).status_code == 202
    limited = client.post(
        "/public/translations",
        json={"segments": [SEGMENT]},
        headers=headers,
    )
    assert limited.status_code == 429
    assert limited.json() == {"detail": "RATE_LIMITED"}
    assert "retry-after" in limited.headers
    assert bridge.public_rate_limiter.events == {}


def test_translation_busy_is_429_and_refunded(client: TestClient) -> None:
    headers = {"X-Forwarded-For": "198.51.100.32"}
    with bridge.translation_manager.lock:
        bridge.translation_manager.active_job_id = "active"
    for _ in range(3):
        response = client.post(
            "/public/translations",
            json={"segments": [SEGMENT]},
            headers=headers,
        )
        assert response.status_code == 429
        assert response.json() == {"detail": "BUSY"}


def test_translation_job_retries_once_and_preserves_alignment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = [
        {
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": 1.0,
            "original": "BTC is worth 1 million dollars.",
        },
        {
            "speaker": "Speaker 2",
            "start": 1.0,
            "end": 2.0,
            "original": "ETH is at 20.",
        },
        {
            "speaker": "Speaker 3",
            "start": 2.0,
            "end": 3.0,
            "original": "See https://example.com for SOL.",
        },
    ]
    retry_calls: list[list[int]] = []

    def fake_translate(
        items: list[dict[str, object]],
        api_key: str,
        *,
        retry: bool = False,
        scheduler: TranslationRateLimitScheduler | None = None,
    ) -> dict[int, str]:
        ids = [int(item["id"]) for item in items]
        if retry:
            retry_calls.append(ids)
            return {
                item_id: (
                    "ETHは20です。"
                    if item_id == 1
                    else "https://example.comをご覧ください。"
                )
                for item_id in ids
            }
        return {
            0: "BTCは100万ドルです。",
            1: "ETHは21です。",
            2: "https://example.comをご覧ください。",
        }

    monkeypatch.setattr(translation_jobs, "translate_batch", fake_translate)
    response = client.post(
        "/public/translations",
        json={"segments": segments},
        headers={"X-Forwarded-For": "198.51.100.33"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", job_id)
    for _ in range(200):
        state = client.get(f"/public/translations/{job_id}").json()
        if state["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    result = client.get(f"/public/translations/{job_id}/result").json()["result"]
    assert state["status"] == "completed"
    assert retry_calls == [[1, 2]]
    assert result["translated_segments"] == 3
    assert result["missing_translations"] == 0
    assert result["alignment"] is True
    assert result["warnings_before_retry"] == 2
    assert result["number_warnings_before_retry"] == 1
    assert result["number_warnings_after_retry"] == 0
    assert result["ticker_warnings_before_retry"] == 1
    assert result["ticker_warnings_after_retry"] == 1
    assert result["remaining_warnings"] == 1
    assert result["segments"][0] == {
        **segments[0],
        "translation": "BTCは100万ドルです。",
        "translation_warning": False,
    }
    assert result["segments"][2]["translation_warning"] is True
    assert result["cleanup"] is True


def test_range_result_preserves_metadata_and_original_index(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "speaker": "Speaker",
        "start": 20.0,
        "end": 21.0,
        "original": "BTC is at 20.",
        "index": 20,
    }

    def fake_translate(
        items: list[dict[str, object]],
        api_key: str,
        *,
        retry: bool = False,
        scheduler: TranslationRateLimitScheduler | None = None,
    ) -> dict[int, str]:
        return {int(items[0]["id"]): "BTCは20です。"}

    monkeypatch.setattr(translation_jobs, "translate_batch", fake_translate)
    job = bridge.translation_manager.submit(
        [source],
        result_context={
            "start_index": 20,
            "count": 1,
            "total_segments": 289,
            "has_more": True,
            "next_index": 21,
        },
    )
    job_id = str(job["job_id"])
    for _ in range(200):
        state = client.get(f"/public/translations/{job_id}").json()
        if state["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    result = client.get(f"/public/translations/{job_id}/result").json()["result"]
    assert state["status"] == "completed"
    assert result["start_index"] == 20
    assert result["count"] == 1
    assert result["total_segments"] == 289
    assert result["has_more"] is True
    assert result["next_index"] == 21
    assert result["segments"][0]["index"] == 20
    assert result["segments"][0]["original"] == source["original"]
    assert result["segments"][0]["translation_warning"] is False
    assert state["batch_total"] == 1
    assert state["batch_completed"] == 1
    assert state["updated_at"] is not None


def test_orphan_processing_future_becomes_failed() -> None:
    manager = translation_jobs.TranslationJobManager(
        ttl_seconds=600,
        batch_segments=10,
        batch_characters=2000,
    )
    job = manager._reserve()
    job_id = str(job["job_id"])
    manager._update(job_id, status="processing", stage="translating")
    future: Future[None] = Future()
    future.set_exception(RuntimeError("worker stopped"))
    with manager.lock:
        manager.futures[job_id] = future
    state = manager.get(job_id)
    assert state is not None
    assert state["status"] == "failed"
    assert state["stage"] == "failed"
    assert state["public_error_code"] == "INTERNAL_ERROR"


def test_range_translation_job_timeout_becomes_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = type("Clock", (), {"now": 0.0})()

    def sleep(seconds: float) -> None:
        clock.now += seconds

    def scheduler_factory(**kwargs: object) -> TranslationRateLimitScheduler:
        return TranslationRateLimitScheduler(
            sleeper=sleep,
            clock=lambda: clock.now,
            wall_clock=lambda: clock.now,
            deadline=1.0,
            telemetry=kwargs.get("telemetry"),  # type: ignore[arg-type]
        )

    def fake_translate(
        items: list[dict[str, object]],
        api_key: str,
        *,
        retry: bool = False,
        scheduler: TranslationRateLimitScheduler | None = None,
    ) -> dict[int, str]:
        assert scheduler is not None
        scheduler._wait(1200, reason="token_reset")
        return {}

    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setattr(
        translation_jobs,
        "TranslationRateLimitScheduler",
        scheduler_factory,
    )
    monkeypatch.setattr(translation_jobs, "translate_batch", fake_translate)
    manager = translation_jobs.TranslationJobManager(
        ttl_seconds=600,
        batch_segments=10,
        batch_characters=2000,
    )
    job = manager.submit([SEGMENT], timeout_seconds=1)
    job_id = str(job["job_id"])
    for _ in range(200):
        state = manager.get(job_id)
        if state and state["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert state is not None
    assert state["status"] == "failed"
    assert state["public_error_code"] == "TRANSLATION_TIMEOUT"
    assert state["waiting"] is False


def test_translation_alignment_failure_splits_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = translation_jobs.TranslationJobManager(
        ttl_seconds=600,
        batch_segments=30,
        batch_characters=6000,
    )
    calls: list[list[int]] = []

    def fake_translate(
        items: list[dict[str, object]],
        api_key: str,
        *,
        retry: bool = False,
        scheduler: TranslationRateLimitScheduler | None = None,
    ) -> dict[int, str]:
        ids = [int(item["id"]) for item in items]
        calls.append(ids)
        if len(items) > 1:
            from render_bridge_app.errors import BridgeError

            raise BridgeError("GROQ_TRANSLATION_ALIGNMENT", "safe")
        return {ids[0]: "訳"}

    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setattr(translation_jobs, "translate_batch", fake_translate)
    result = manager._translate_aligned(
        [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}]
    )
    assert result == {0: "訳", 1: "訳"}
    assert calls == [[0, 1], [0], [1]]


def test_translation_results_expire_and_unknown_is_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge.translation_manager, "ttl_seconds", 1)
    with bridge.translation_manager.lock:
        bridge.translation_manager.jobs["expired"] = {
            "job_id": "expired",
            "status": "completed",
            "stage": "completed",
            "created_at": time.time() - 10,
            "finished_at": time.time() - 2,
        }
    assert client.get("/public/translations/expired").status_code == 404
    assert client.get("/public/translations/missing/result").status_code == 404


@pytest.mark.parametrize(
    ("internal", "public"),
    [
        ("GROQ_TRANSLATION_RATE_LIMIT", "GROQ_RATE_LIMITED"),
        ("GROQ_TRANSLATION_ALIGNMENT", "TRANSLATION_ALIGNMENT_FAILED"),
        ("GROQ_TRANSLATION_ERROR", "GROQ_TRANSLATION_FAILED"),
        ("CODE_ERROR", "INTERNAL_ERROR"),
    ],
)
def test_translation_error_classification(internal: str, public: str) -> None:
    assert translation_jobs.public_translation_error_code(internal) == public
