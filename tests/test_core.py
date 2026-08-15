import wave
from pathlib import Path

import pytest

from app.database.store import Store
from app.services.audio import validate_upload
from app.services.downloader import normalize_x_url
from app.services.errors import InvalidSourceError
from app.services.exports import render, timestamp
from app.services.transcriber import MAX_CHUNK_SECONDS, split_wav


def test_normalize_supported_x_urls():
    assert (
        normalize_x_url("https://twitter.com/i/spaces/abc?utm_source=x")
        == "https://x.com/i/spaces/abc"
    )
    assert (
        normalize_x_url("https://x.com/i/broadcasts/123")
        == "https://x.com/i/broadcasts/123"
    )


@pytest.mark.parametrize(
    "url", ["https://youtube.com/watch?v=x", "file:///secret", "https://x.com/home"]
)
def test_reject_non_space_url(url):
    with pytest.raises(InvalidSourceError):
        normalize_x_url(url)


def test_upload_validation():
    assert validate_upload("talk.MP3", "audio/mpeg") == ".mp3"
    with pytest.raises(InvalidSourceError):
        validate_upload("payload.exe", "application/octet-stream")


def test_long_wav_is_split_without_loading_it_all(tmp_path: Path):
    audio = tmp_path / "long.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(1)
        output.writeframes(b"\0\0" * (MAX_CHUNK_SECONDS * 2 + 1))

    chunks, duration = split_wav(audio)

    assert duration == MAX_CHUNK_SECONDS * 2 + 1
    assert [offset for _, offset in chunks] == [0, 1200, 2400]
    assert all(path.exists() for path, _ in chunks)


def test_timestamp_and_exports():
    result = {
        "segments": [
            {
                "speaker": "Speaker A",
                "start": 74.2,
                "end": 81.5,
                "original": "Hello",
                "translation_ja": "こんにちは",
            }
        ]
    }
    assert timestamp(74.2) == "00:01:14,200"
    srt, mime = render(result, "srt", "both")
    assert "00:01:14,200 --> 00:01:21,500" in srt
    assert "こんにちは" in srt
    assert mime == "application/x-subrip"


def test_store_round_trip_and_speaker_rename(tmp_path: Path):
    store = Store(tmp_path / "test.db")
    store.create_job("job1", "file", source_path="audio.wav")
    transcript_id = store.save_transcript(
        "key",
        {
            "title": "Demo",
            "detected_language": "en",
            "duration": 2.0,
            "segments": [
                {
                    "speaker": "Speaker A",
                    "start": 0,
                    "end": 2,
                    "original": "Hello",
                    "translation_ja": "",
                }
            ],
        },
    )
    store.update_job("job1", status="completed", transcript_id=transcript_id)
    store.rename_speakers(transcript_id, {"Speaker A": "Alice"})
    result = store.get_transcript(transcript_id)
    assert result["segments"][0]["speaker"] == "Alice"
    assert store.find_cached("key") == transcript_id
