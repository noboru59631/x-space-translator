# Cloud Run worker PoC

This directory contains the isolated CPU worker for the Cloud Run proof of concept.
The existing local application remains the primary product and does not depend on this
worker.

## API

All job endpoints require `Authorization: Bearer <WORKER_API_KEY>`.
`GET /health` is public and reports only dependency availability.

- `POST /jobs` with `{"url":"https://x.com/i/spaces/..."}`
- `POST /jobs/file` with a small audio file (PoC testing only, maximum 25 MB)
- `GET /jobs/{job_id}` for status, stage and progress
- `GET /jobs/{job_id}/result` for the completed transcript

Only X/Twitter Space and broadcast URLs are accepted. YouTube, arbitrary hosts,
localhost, private-network URLs and non-HTTP schemes are rejected by the shared local
URL validator. Downloads do not use cookies.

The PoC accepts only one active job. A second submission receives HTTP 429. Results
live in process memory and temporary audio lives under `/tmp/x-space-worker`; both are
lost when an instance restarts. Audio and converted files are deleted in a `finally`
block after success or failure.

## Local Docker run

Build from the repository root:

```text
docker build -f cloud/worker/Dockerfile -t x-space-translator-worker:poc .
docker run --rm -p 8080:8080 -e WORKER_API_KEY=local-test-key \
  -e WHISPER_CPU_THREADS=1 x-space-translator-worker:poc
```

The image downloads the `faster-whisper` base model at first use. This keeps the image
and build smaller, but adds a first-request delay and requires model-network access.
An instance restart may require another download because Cloud Run's writable file
system is ephemeral. Embedding the model during the image build is the alternative;
it gives predictable cold starts at the cost of a substantially larger image and build.
Runtime download is used for this PoC so model choice can be changed without rebuilding.

## Proposed Cloud Run settings

- Region: `asia-northeast1`
- CPU: 1
- Memory: 4 GiB
- Request concurrency: 1
- Minimum instances: 0
- Maximum instances: 1
- Request timeout: 1200 seconds
- CPU allocation: always allocated (`--no-cpu-throttling`)
- Secret: `WORKER_API_KEY` from Secret Manager

The POST response returns before background transcription completes. Cloud Run must
therefore keep CPU allocated outside request handling. Even with that setting, a
minimum instance count of zero means the instance can stop and erase in-memory job
state. This is an explicit PoC limitation; production needs durable job and result
storage.

Example after a project, billing, Artifact Registry and a Secret Manager secret have
already been configured by the account owner:

```text
gcloud run deploy x-space-translator-worker \
  --image asia-northeast1-docker.pkg.dev/PROJECT/REPOSITORY/IMAGE:TAG \
  --region asia-northeast1 --cpu 1 --memory 4Gi \
  --concurrency 1 --min 0 --max 1 --timeout 1200 \
  --no-cpu-throttling \
  --set-env-vars WHISPER_MODEL=base,WHISPER_COMPUTE_TYPE=int8,WHISPER_CPU_THREADS=1 \
  --set-secrets WORKER_API_KEY=worker-api-key:1 \
  --allow-unauthenticated
```

`--allow-unauthenticated` makes the HTTP endpoint reachable, while the application
still requires its Bearer key. Google IAM authentication is preferable for a future
production service. Budget alerts help with visibility but are not a hard spending cap.
Do not run the deployment command until the account owner has reviewed billing and
limits.
