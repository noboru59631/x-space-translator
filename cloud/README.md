# Cloud PoC

This directory is isolated from the local Windows application. The local `app/`, `setup.bat`, `start.bat`, Bankr integration and exports continue to work unchanged.

## Architecture

1. `cloud/vercel` validates an X Spaces URL and submits it to the worker.
2. `cloud/worker` returns a `job_id` immediately.
3. The worker downloads audio with the existing yt-dlp code, converts it with FFmpeg and transcribes it with faster-whisper `base` on CPU/int8.
4. Poll `GET /jobs/{job_id}` until `status` is `completed` or `failed`.
5. Audio and WAV files are deleted after the job. Transcript JSON remains only in worker memory for a limited time.

## Local Worker

```powershell
docker build -f cloud/worker/Dockerfile -t x-space-translator-worker:poc .
docker run --rm -p 127.0.0.1:7860:7860 x-space-translator-worker:poc
```

Submit an X Space:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/transcribe `
  -ContentType application/json `
  -Body '{"url":"https://x.com/i/spaces/1qxvvvQBRXQxB/peek"}'
```

Upload a short audio file:

```powershell
curl.exe -F "file=@sample.wav" http://127.0.0.1:7860/transcribe/file
```

## Vercel

Set `WORKER_API_URL` and the same optional `WORKER_API_TOKEN` used by the Worker. Vercel performs no audio, FFmpeg or model processing.

## Hugging Face Docker Space

Build the Space from the repository root using `cloud/worker/Dockerfile`. For a dedicated Space repository, copy `cloud/worker/README_HF.md` to `README.md`, preserve the root `app/` and `cloud/` directories, and copy the Dockerfile to the repository root.
