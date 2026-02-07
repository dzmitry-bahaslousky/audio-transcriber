# Audio Transcriber

Audio transcription service built with [FastAPI](https://fastapi.tiangolo.com/) and [WhisperX](https://github.com/m-bain/whisperX). Supports automatic speech recognition, word-level alignment, and speaker diarization.

## Features

- Speech-to-text transcription via WhisperX
- Optional word-level timestamps
- Optional speaker diarization (who spoke when)
- Configurable language detection or language override
- Automatic GPU/CPU detection
- Idle model unloading to free memory

## Prerequisites

- Python 3.11+
- ffmpeg
- [HuggingFace account](https://huggingface.co/) with an access token (required for diarization models)

## Local Setup

1. **Install ffmpeg** (if not already installed):

   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**

   For CPU-only (development):
   ```bash
   pip install -r requirements-cpu.txt
   pip install -r requirements.txt
   ```

   For CUDA GPU:
   ```bash
   pip install -r requirements-cuda.txt
   pip install -r requirements.txt
   ```

4. **Set environment variables:**

   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

5. **Start the server:**

   ```bash
   uvicorn app.main:api --reload
   ```

   The API is available at http://localhost:8000 with interactive docs at http://localhost:8000/docs.

## Docker Setup

### CPU

```bash
export HF_TOKEN=your_huggingface_token
docker compose -f docker-compose.cpu.yml up --build
```

### CUDA GPU

```bash
docker build -f Dockerfile.cuda -t audio-transcriber:cuda .
docker run -p 8000:8000 \
  --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v model-cache:/home/appuser/.cache/whisper \
  audio-transcriber:cuda
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Usage

Send a POST request with an audio file and a JSON config:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F 'config={"languageCode": "en", "features": {"enableWordLevelTimestamps": true, "diarization": {"enableSpeakerDiarization": true}}}'
```

### Config Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `languageCode` | string | auto-detect | Language code (e.g. `"en"`, `"fr"`) |
| `features.enableWordLevelTimestamps` | bool | `false` | Enable word-level alignment |
| `features.diarization.enableSpeakerDiarization` | bool | `true` | Enable speaker diarization |
| `features.diarization.minSpeakerCount` | int | null | Minimum expected speakers |
| `features.diarization.maxSpeakerCount` | int | null | Maximum expected speakers |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | - | HuggingFace access token |
| `ASR_MODEL` | No | `base` | Model size: tiny, base, small, medium, large-v2, large-v3 |
| `ASR_DEVICE` | No | auto-detect | `cuda` or `cpu` |
| `ASR_COMPUTE_TYPE` | No | `float32` (GPU) / `int8` (CPU) | float32, float16, or int8 |
| `ASR_BATCH_SIZE` | No | `4` | Transcription batch size |
| `ASR_MODEL_CACHE_PATH` | No | `~/.cache/whisper` | Model download location |
| `ASR_MODEL_IDLE_TIMEOUT` | No | `0` | Seconds before unloading idle model (0 = never) |
| `SAMPLE_RATE` | No | `16000` | Audio sample rate in Hz |

## Health Checks

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /health/liveness` | Liveness probe |
| `GET /health/readiness` | Readiness probe (503 if models not loaded) |
