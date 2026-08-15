# X Space Translator

Paste an X Space URL and read the conversation in English and Japanese.

Runs locally on your PC. No paid transcription API required.

**Status:** Beta

**Current release:** `v0.1.0-beta`

X Space Translator downloads an available X Space audio stream—or accepts a local audio/video file—then converts, transcribes, optionally separates speakers, translates to Japanese, and exports the result. The FastAPI interface is intentionally bound to `127.0.0.1` by default.

> X may change its delivery format or restrict access. Direct audio retrieval from a Space URL can therefore fail. When it does, upload an audio file and continue. A failed X download is not automatically an application defect.

## Features

- X-only URL validation for `x.com` / `twitter.com` Spaces and broadcasts
- MP3, WAV, M4A, MP4, and WEBM upload
- Local transcription with faster-whisper and automatic CPU/CUDA selection
- Lightweight, standard, and accurate model presets
- Optional speaker diarization with pyannote.audio
- Local English-to-Japanese translation with Meta M2M100
- English, Japanese, and bilingual views without retranscription
- Speaker renaming reflected in every export
- TXT, SRT, WebVTT, and JSON export
- SQLite job/transcript cache and restart-safe completed results
- Background jobs, progress polling, cancellation, and temporary-file cleanup
- Local-only, responsive web UI and documented FastAPI endpoints

### Core and Optional Features

Core features in this beta are X Space URL/local file input, local faster-whisper transcription, English/Japanese/bilingual display, TXT/SRT/VTT/JSON export, copy, and SQLite caching.

Speaker diarization, cookie-assisted X access, and GPU acceleration are optional beta features. They depend on external credentials, upstream services, hardware, or environment-specific packages and are not required for the core workflow.

## Tested Environment and Example Benchmark

These figures are one real-world example from a single Windows PC, not minimum requirements or guaranteed performance:

- X Space length: about 22 minutes 42 seconds
- URL retrieval: about 16 seconds; this one recording downloaded without cookies
- Transcription: faster-whisper `base`, CPU, `int8`, 176 segments, about 3 minutes 1 second for the pipeline
- Transcription peak private memory: about 2.0 GB
- Translation: [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M), CPU with up to 4 threads, 176/176 segments, about 239.2 seconds
- Translation peak private memory: about 3.21 GB

During one E2E test, the X Space was downloaded successfully without cookies. Some Spaces or future X changes may still require authentication or prevent direct download.

## Screenshots

Screenshots are intentionally not fabricated. Add verified application screenshots here before a public release.

## Requirements

- Windows 10 or 11
- Python 3.11 (64-bit recommended)
- FFmpeg available on `PATH`
- About 3–10 GB of free disk space depending on selected models
- Internet access for the first model download and for X URL retrieval
- NVIDIA GPU is optional; CPU mode works but is slower

## Quick Start (Windows)

1. Download the GitHub ZIP and extract it.
2. Double-click `setup.bat` and wait for installation to complete.
3. Install FFmpeg if the setup window reports that it is missing:

   ```powershell
   winget install Gyan.FFmpeg
   ```

4. Open a new terminal, then double-click `start.bat`.
5. The browser opens at <http://127.0.0.1:8765>.
6. Paste an X Space URL and start transcription. If X retrieval fails, switch to the audio-file tab.

The first transcription or translation downloads models and can take several minutes. Model files are supplied under their respective upstream licenses.

## Manual Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

## CPU Mode

The default `WHISPER_DEVICE=auto` uses CUDA when ctranslate2 can access an NVIDIA GPU and falls back to CPU otherwise. On CPU, use lightweight or standard mode. CPU inference uses `int8` by default to reduce memory pressure.

To force CPU mode, edit `.env`:

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

## GPU Mode

GPU acceleration requires an NVIDIA driver and compatible CUDA/cuDNN libraries for your installed ctranslate2/PyTorch versions. Because compatible versions depend on the GPU and driver, install the appropriate PyTorch build first, then run:

```powershell
.venv\Scripts\python -m pip install -r requirements-gpu.txt
```

Set `WHISPER_DEVICE=cuda` only after CUDA works. If initialization fails, return it to `auto` or `cpu`. High-accuracy mode uses `large-v3` and needs substantially more VRAM.

## FFmpeg

The app checks for `ffmpeg.exe` at runtime and never silently skips conversion. Verify the installation with:

```powershell
ffmpeg -version
```

If Windows cannot find it after installation, open a new terminal or add the FFmpeg `bin` directory to `PATH`.

## Speaker Diarization and Hugging Face Token

Transcription works without a Hugging Face token. The token is needed only for speaker diarization.

1. Accept the terms for the upstream pyannote diarization model on Hugging Face.
2. Create a read token.
3. Install `requirements-gpu.txt`.
4. Put the token only in your local `.env`:

   ```env
   HF_TOKEN=hf_your_token
   ```

If diarization is unavailable or fails, the transcript remains available and all segments use `Speaker A`. The program does not infer real people's names from audio. Rename speakers manually only when you know the mapping.

## X Cookie Setting

Some Spaces may require authenticated browser cookies. Export a Netscape-format cookie file yourself, store it outside the repository if possible, and set its absolute path:

```env
X_COOKIE_FILE=C:\private\x-cookies.txt
```

Cookie content is never logged. Do not commit cookie files, share them, or embed them in source code. Cookies can grant account access; handle them as credentials.

## Configuration

Copy `.env.example` to `.env` and edit local values:

```env
APP_HOST=127.0.0.1
APP_PORT=8765
WHISPER_MODEL=small
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
HF_TOKEN=
X_COOKIE_FILE=
MAX_UPLOAD_MB=2048
TEMP_DIR=./temp
DATA_DIR=./data
CORS_ORIGINS=http://127.0.0.1:8765,http://localhost:8765
```

Do not use `APP_HOST=0.0.0.0` unless you understand the network exposure. CORS does not replace authentication or a firewall, and `*` is deliberately not the default.

## API

Interactive API documentation is available locally at <http://127.0.0.1:8765/docs>.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Check FFmpeg, Whisper, GPU, diarization, and translation |
| POST | `/api/transcribe/url` | Queue an X Space URL |
| POST | `/api/transcribe/file` | Upload and queue an audio/video file |
| POST | `/api/translate` | Translate a completed transcript |
| GET | `/api/jobs/{job_id}` | Poll job stage and progress |
| GET | `/api/jobs/{job_id}/result` | Read the structured result |
| PUT | `/api/jobs/{job_id}/speakers` | Rename speaker labels |
| GET | `/api/jobs/{job_id}/export/{format}` | Download TXT/SRT/VTT/JSON |
| POST | `/api/jobs/{job_id}/cancel` | Request cancellation |

Cancellation is cooperative: the current native inference call may finish before cancellation is observed.

## Docker (Optional)

Windows beginners should prefer `setup.bat`. Docker CPU mode is available for experienced users:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open <http://127.0.0.1:8765>. NVIDIA container configuration is environment-specific and is not enabled by default.

## Tests

Core tests do not download models or media:

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m ruff check app tests
.venv\Scripts\python -m pytest -q
```

Real transcription requires FFmpeg plus locally downloaded models. Test only media you are allowed to process. The project does not ship fake “successful” transcript data.

## Troubleshooting

**X Space audio could not be retrieved**  
Check that the URL points to `/i/spaces/` or `/i/broadcasts/`, that a recording is publicly playable, and whether a cookie is required. X delivery changes can still prevent retrieval. Upload a legally obtained audio file instead.

During one E2E test, the X Space was downloaded successfully without cookies. Some Spaces or future X changes may still require authentication or prevent direct download.

**FFmpeg was not found**  
Run `winget install Gyan.FFmpeg`, open a new terminal, and verify `ffmpeg -version`.

**Out of memory**  
Before transcribing longer Spaces, close memory-heavy applications such as browsers, games, video editors, or other AI applications. If available memory is very low, restart Windows or use Lightweight mode. You can also force CPU `int8` or process a shorter file. The app converts to mono 16 kHz audio and streams uploads to disk, but model inference still needs memory.

In one test on this PC, processing failed with only about 0.55 GB of available memory and succeeded after restart with about 5.42 GB available; observed peak private memory was about 2.0 GB for transcription and 3.21 GB for translation. These are observations, not minimum requirements.

**The first run appears slow**  
faster-whisper and translation models download on first use. Later runs reuse the local model cache.

**Speaker diarization is unavailable**  
Install `requirements-gpu.txt`, accept the pyannote model terms, and set `HF_TOKEN`. Transcription itself does not require diarization.

## View transcripts in Bankr

After transcription, save the JSON file and open it in the Bankr Transcript Viewer.

Bankr Viewer supports:

- English
- Japanese
- EN + JA
- Search
- Speaker filtering

Transcript JSON is currently loaded locally in the browser and is not automatically uploaded to Bankr. Bankr is an optional viewer; transcription, translation, and exports do not depend on it.

## Privacy

Audio processing is designed to run on your PC. Uploaded media is written only to a per-job temporary directory and removed after processing. Results and job state are stored in `data/x_space_translator.db`. Initial model downloads and X URL retrieval require external network connections; media is not otherwise deliberately sent to a paid transcription or translation API.

Delete the local `data` directory if you want to erase stored transcripts. It is excluded from Git.

## Limitations

- X URL extraction is best-effort and depends on X and yt-dlp behavior.
- Only X Spaces/broadcast URLs are accepted; YouTube is intentionally unsupported.
- English-to-Japanese is the primary translation path. Other detected languages may transcribe, but translation quality or support is not guaranteed.
- Speaker labels are anonymous and approximate. They do not identify real people.
- Very long recordings can take hours on CPU.
- Cancellation cannot interrupt every third-party native call immediately.

## Responsible Use and Copyright

Use only content you have the right or permission to transcribe and translate. Follow the content's terms of use, copyright, privacy, and applicable law. This tool does not grant rights to download, reproduce, publish, or redistribute a Space or its transcript.

## GitHub Issues

- **Bug Report:** include Windows/Python versions, CPU/GPU, exact stage, sanitized logs, and reproduction steps. Never attach cookies or tokens.
- **Feature Request:** describe the workflow, expected behavior, and why it benefits local transcription.

## License

Application source is released under the [MIT License](LICENSE). Dependencies and downloaded models retain their own licenses and terms.

---

# 日本語

X SpaceのURLを貼るだけで、会話を文字起こしし、日本語で読めるローカルアプリです。文字起こし処理はあなたのPC上で実行され、有料の文字起こしAPIは不要です。

**状態:** Beta

**現在のリリース:** `v0.1.0-beta`

## できること

- X Space URLまたはMP3/WAV/M4A/MP4/WEBMを入力
- faster-whisperによるローカル文字起こし
- CPU/GPU自動判定、軽量・標準・高精度モード
- 任意の話者分離と手動の話者名変更
- 英語から日本語へのローカル翻訳
- English / 日本語 / EN + JA 表示
- TXT / SRT / VTT / JSON保存と全文コピー
- 長時間処理向けのバックグラウンドジョブ、進捗表示、SQLiteキャッシュ

## コア機能と任意機能

このBeta版のコア機能は、X Space URL／ローカル音声の入力、faster-whisperによるローカル文字起こし、英語・日本語・併記表示、コピー、TXT／SRT／VTT／JSON出力、SQLiteキャッシュです。

話者分離、Cookieを利用したXへのアクセス、GPU高速化は任意のBeta機能です。外部認証情報、上流サービス、ハードウェア、環境依存パッケージに左右され、コア機能には必要ありません。

## 実測環境とベンチマーク例

以下は1台のWindows PCで得た実例であり、最低要件や処理速度を保証するものではありません。

- X Spaceの長さ：約22分42秒
- URL取得：約16秒。この録音はCookieなしで取得成功
- 文字起こし：faster-whisper `base`、CPU、`int8`、176区間、パイプライン約3分1秒
- 文字起こし時の最大プライベートメモリ：約2.0 GB
- 翻訳：[`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)、CPU最大4スレッド、176/176区間、約239.2秒
- 翻訳時の最大プライベートメモリ：約3.21 GB

1回のE2Eテストでは、CookieなしでX Spaceの音声取得に成功しました。ただし、別のSpaceや今後のXの変更では、認証が必要になったり、直接取得できなくなったりする可能性があります。

## Windowsでの使い方

1. GitHubのZIPをダウンロードして展開します。
2. `setup.bat`をダブルクリックします。
3. FFmpegがないと表示されたら、PowerShellで`winget install Gyan.FFmpeg`を実行します。
4. `start.bat`をダブルクリックします。
5. 開いたブラウザでX Space URLを貼り、文字起こしを開始します。
6. Xから取得できない場合は「音声ファイル」に切り替えて続行します。

話者分離を使わない場合、Hugging Face Tokenは不要です。CookieやTokenは`.env`にだけ保存し、GitHubへcommitしないでください。

## Bankrで文字起こしを見る

文字起こし完了後にJSONファイルを保存し、Bankr Transcript Viewerで開くことができます。

Bankr Viewerでは以下を利用できます。

- English
- 日本語
- EN + JA
- 検索
- Speakerフィルター

Transcript JSONは現在ブラウザ内でローカルに読み込まれ、Bankrへ自動アップロードされません。Bankrは任意のViewerであり、文字起こし・翻訳・各種出力はBankrに依存しません。

## 注意事項

X側の仕様変更やアクセス制限により、Space URLから直接音声を取得できない場合があります。その場合は音声ファイルアップロードを利用してください。確認していない処理を「成功」と表示するデモデータは含めていません。

長めのSpaceを文字起こしする前に、ブラウザ、ゲーム、動画編集ソフト、ほかのAIアプリなど、メモリを多く使うアプリを閉じてください。空きメモリが非常に少ない場合は、Windowsを再起動するか軽量モードを利用してください。このPCでの1回の実測では、空き約0.55 GBでは失敗し、再起動後の空き約5.42 GBでは成功しました。文字起こし時の最大プライベートメモリは約2.0 GB、翻訳時は約3.21 GBでしたが、これは最低要件ではありません。

音声処理は原則として利用者自身のPC上で行われます。ただし、初回のモデル取得とX URLからの音声取得には外部通信が必要です。文字起こし・翻訳する権利または許可のあるコンテンツだけを利用し、利用規約・著作権・プライバシー・法令を守ってください。
