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
JSON, a conservative 4,096-token completion cap, and bounded batches. The
result preserves those fields and adds
`translation` and `translation_warning`.

Groq HTTP 429 responses are retried with the provider's reset guidance (or a
bounded backoff) because a full transcript can span multiple model requests.
This transport retry is separate from the one preservation-correction retry.
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
