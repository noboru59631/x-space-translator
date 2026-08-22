# X Space Translator Render Bridge PoC

This isolated service downloads an X Space, copies its AAC audio into a valid
M4A container, validates it with ffprobe, and streams the file to Groq Whisper.
It does not install faster-whisper, M2M100, PyTorch, Transformers, or any model
weights.

## Flow

1. Accept an allowlisted X Space or broadcast URL.
2. Download to a random job directory with yt-dlp and no cookies.
3. Remux the first audio stream with system FFmpeg and `-c:a copy`.
4. Validate file existence, size, positive duration, container, and AAC codec.
5. Stream the M4A file to Groq `whisper-large-v3-turbo`.
6. Return Bankr Viewer-compatible English transcript JSON.
7. Delete the complete job directory in a `finally` block.

The service does not re-encode audio. If stream copy cannot produce a valid
AAC/M4A file, the job fails before contacting Groq.

## API

`GET /health` is public and returns only dependency availability.

Bankr can use these secretless, X-Spaces-only endpoints:

- `POST /public/jobs`
- `GET /public/jobs/{job_id}`
- `GET /public/jobs/{job_id}/result`

The public creation endpoint accepts only allowlisted X/Twitter Space and
broadcast URLs. It rejects YouTube, arbitrary hosts and schemes, localhost,
private IP targets, and oversized request bodies. Public jobs have random UUID
identifiers and cannot be used to read jobs created through authenticated
routes.

The default public limit is two accepted jobs per source IP per 10 minutes.
Only one job can run across the complete service at a time; another submission
receives HTTP 429 `BUSY` and does not consume its IP allowance. Public audio is
limited to two hours and is rejected before Groq upload when the validated
duration is longer. These controls are configurable with
`PUBLIC_RATE_LIMIT_JOBS`, `PUBLIC_RATE_LIMIT_WINDOW_SECONDS`, and
`PUBLIC_MAX_AUDIO_SECONDS`.

### Public Japanese translation jobs

Bankr can translate an existing Viewer transcript without receiving the Groq
secret:

- `POST /public/translations`
- `GET /public/translations/{translation_job_id}`
- `GET /public/translations/{translation_job_id}/result`

The POST body contains Viewer-compatible `segments` with `speaker`, `start`,
`end`, and `original`. It returns HTTP 202 immediately. Translation runs in a
single background worker with Groq `openai/gpt-oss-120b`, strict structured
JSON, a translation-sized dynamic completion cap, and bounded batches. The
result preserves those fields and adds
`translation` and `translation_warning`.

Free Plan batches are limited to 10 segments and 2,000 source characters. A
token-aware scheduler retains only numeric rate-limit header values, uses the
reported remaining TPM and reset duration before sending the next batch, and
keeps a conservative local 60-second token window. If those headers are
missing, requests are spaced by 20 seconds. Groq HTTP 429 responses use
`retry-after`, then the token reset duration, then bounded exponential backoff.
The retry count remains finite. This transport retry is separate from the one
preservation-correction retry.
If structured output ever omits or misaligns batch IDs, that batch is split and
retried in smaller halves until alignment is exact or a single segment fails.

Numbers, URLs, BTC, ETH, SOL, XRP, USDT, and USDC are compared between each
original and translation. A mismatched segment is translated one more time;
only a mismatch that remains after that retry sets `translation_warning`.
Equivalent full-width digits and common English/Japanese scale forms such as
`1 million` and `100万` are normalized to reduce false warnings.

Translation jobs have a separate per-IP allowance of two accepted jobs per 10
minutes. Requests are limited to 500 segments, 120,000 original characters,
and a 1 MiB JSON body by default. Transcription and translation submissions
share a service-wide admission check so only one heavy operation is accepted
at a time. Job IDs are random UUIDs and results expire from memory after the
same `JOB_TTL_SECONDS` interval. No transcript or translation is written to a
database or temporary file, and neither text is logged.

When transcription was created through `POST /public/jobs`, the completed
in-memory transcript can also be translated without sending its segments back:

- `POST /public/jobs/{transcript_job_id}/translations`

The optional request body selects a range and defaults to the first 20
segments:

```json
{"start_index": 0, "count": 20}
```

`count` must be between 1 and 25. Only that slice is submitted to the existing
translation worker; the bridge never advances automatically to the next
range. Each translated segment includes its original zero-based `index`.
The result also contains `start_index`, the actual returned `count`,
`total_segments`, `has_more`, and `next_index`. A final partial range returns
`has_more: false` and `next_index: null`.

The response contains the translation `job_id`, which is polled through the
existing `GET /public/translations/{job_id}` and
`GET /public/translations/{job_id}/result` endpoints. Repeated requests reuse
the existing queued, processing, or completed translation job for the same
range instead of starting another Groq request. A different range can be
started only after the active translation has finished and requires a new
explicit user action.

This Job-ID translation route works only while the completed Transcript Job is
still within its 30-minute TTL and remains in the same running Render process.
Render restarts, redeploys, and TTL expiry remove the in-memory transcript, so
the route then returns `TRANSCRIPT_JOB_NOT_FOUND`. No database or Redis storage
is used.

Translation requests use a strict rolling 60-second token budget for Groq's
Free Plan. The provider limit is treated as 8,000 TPM, while the bridge uses at
most 6,000 estimated or measured tokens and keeps 2,000 tokens in reserve.
Batches remain capped at 10 segments and 2,000 characters, and are reduced
further to target about 1,250 total input/output tokens per request. Successful
usage values replace estimates in the rolling window. When available, Groq's
remaining-token and reset headers take priority; otherwise the bridge applies
conservative fixed pacing. Translation must be polled as an asynchronous job.
Full-transcript automatic translation is intentionally not performed by the
Job-ID route on the Free Plan.

Provider-directed waits are split into at most
`TRANSLATION_MAX_SINGLE_WAIT_SECONDS` (60 seconds by default), so a long reset
header cannot leave a job in one opaque sleep. Range jobs have a
`TRANSLATION_JOB_TIMEOUT_SECONDS` deadline (600 seconds by default). While a
job is running, its status response exposes only safe operational telemetry,
including batch progress, request counts, wait reason, bounded wait duration,
and heartbeat timestamps. A timed-out job becomes `failed` with
`TRANSLATION_TIMEOUT`; transcript and translation text are not included in
telemetry.

The following endpoints require
`Authorization: Bearer <BRIDGE_API_KEY>`:

- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `POST /jobs`
- `POST /transcribe` (job-creation alias)
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

Example request:

```json
{"url": "https://x.com/i/spaces/1qxvvvQBRXQxB/peek"}
```

`POST /api/jobs` returns HTTP 202 immediately with a random `job_id`; download,
FFmpeg, and Groq work run on the single background worker. Poll the matching
status endpoint until it reaches `completed`, then read the result endpoint.
Progress is `null` because the bridge does not invent a percentage it cannot
measure accurately. A second submission while one job is active receives HTTP
429.

Completed and failed API results remain in memory for `JOB_TTL_SECONDS` (30
minutes by default). Audio and the complete job temp directory are deleted as
soon as processing finishes; only transcript JSON remains in the job store.
The legacy `/jobs` and `/transcribe` routes remain available for compatibility.

Jobs and results are held only in memory. They disappear on a Render restart,
deploy, or free-instance spin-down. Production should use an external durable
job store. The public rate limiter is also in memory and resets on restart.
This is a PoC limitation.

## Local Docker

Build from the repository root:

```text
docker build -f cloud/render-bridge/Dockerfile \
  -t x-space-translator-render-bridge:poc .
```

Run with secrets supplied only at runtime:

```text
docker run --rm -p 10000:10000 \
  -e BRIDGE_API_KEY=<random-secret> \
  -e GROQ_API_KEY=<groq-secret> \
  x-space-translator-render-bridge:poc
```

## Groq limit

Groq's current free-tier speech-to-text upload limit is 25 MB. The bridge
enforces that limit after remuxing and before upload. It does not fall back to
lossy re-encoding or chunking in this PoC.

Audio is sent to Groq for transcription. The bridge does not persist audio or
transcripts, but users must account for Groq's own data handling terms.

## Render Free

The included `render.yaml` explicitly requests a Free Web Service, uses
`/health`, generates `BRIDGE_API_KEY`, and prompts for `GROQ_API_KEY`
without committing either value.

Create a Blueprint from this repository and select
`cloud/render-bridge/render.yaml` if Render does not use the root default.
Review that the selected plan is **Free** before creating the service.

Current Render documentation lists Free Web Services at 512 MB RAM and 0.1 CPU,
and states that they spin down after 15 minutes without inbound traffic. The
first request after sleep therefore includes cold-start delay. No persistent
disk is used.

Deployment requires this directory to exist in the Git commit visible to
Render. Do not deploy from the older public commit, because it does not contain
the bridge.
