from __future__ import annotations

import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from render_bridge_app import main as bridge
from render_bridge_app import translation_client, translation_jobs
from render_bridge_app.main import app
from render_bridge_app.translation_validation import validate_preservation

SEGMENT = {
    "speaker": "Speaker",
    "start": 0,
    "end": 10,
    "original": "BTC reached 1 million dollars at https://example.com/a.",
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
    monkeypatch.setattr(translation_client.time, "sleep", waits.append)
    result = translation_client.translate_batch(
        [{"id": 1, "text": "Success"}],
        "secret",
    )
    assert result == {1: "成功"}
    assert waits == [1.5]


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
