import os

import torch
from dotenv import load_dotenv

load_dotenv()

# Fix PyTorch 2.6+ weights_only=True default breaking WhisperX model loading
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")

class Config:
    ASR_MODEL = os.getenv("ASR_MODEL", "base")

    ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    ASR_MODEL_CACHE_PATH = os.getenv("ASR_MODEL_CACHE_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        raise Exception("HF_TOKEN environment variable is not set")

    ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float32" if torch.cuda.is_available() else "int8")
    if ASR_COMPUTE_TYPE not in {"float32", "float16", "int8"}:
        raise ValueError("Invalid ASR_COMPUTE_TYPE. Choose 'float32', 'float16', or 'int8'.")

    ASR_MODEL_IDLE_TIMEOUT = int(os.getenv("ASR_MODEL_IDLE_TIMEOUT", 0))

    ASR_BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", 4))

    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))


