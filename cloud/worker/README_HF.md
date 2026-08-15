---
title: X Space Translator Worker PoC
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# X Space Translator Worker PoC

CPU-only proof-of-concept worker using yt-dlp, FFmpeg and faster-whisper base/int8.

This service stores transcripts only in process memory. Downloaded audio and converted WAV files are removed after every job.

Set `WORKER_API_TOKEN` as a Hugging Face Space secret before exposing the API.
