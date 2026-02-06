import gc
import time
from threading import Thread, Lock
from typing import Any, TypedDict

import torch

import whisperx
from whisperx.diarize import DiarizationPipeline

from app.audio_utils import load_audio
from app.log_config import get_logger
from app.config import Config
from app.model import ConfigRequest

logger = get_logger(__name__)


class ModelDict(TypedDict, total=False):
    """Type definition for the model dictionary."""
    asr: Any  # whisperx.Model type not publicly exposed
    diarize: DiarizationPipeline
    alignment: Any


class WhisperxASR:
    _model: ModelDict = {}

    _model_lock = Lock()
    _last_activity_time = time.time()

    def load_model(self):
        logger.info(f"Loading ASR model: '{Config.ASR_MODEL}' with device: '{Config.ASR_DEVICE}'")
        self._model['asr'] = whisperx.load_model(
            Config.ASR_MODEL,
            device=Config.ASR_DEVICE,
            compute_type=Config.ASR_COMPUTE_TYPE,
            download_root=Config.ASR_MODEL_CACHE_PATH
        )
        logger.info("Loaded ASR model")

        logger.info("Loading diarization model")
        self._model['diarize'] = DiarizationPipeline(
            use_auth_token=Config.HF_TOKEN,
            device=Config.ASR_DEVICE
        )

        Thread(target=self._monitor_idleness, daemon=True).start()

    def transcribe(self, audio, config_request: ConfigRequest):
        logger.debug(f"Transcribing request: {config_request}")
        self._ensure_model_loaded()

        options_dict = {"batch_size": Config.ASR_BATCH_SIZE}
        if config_request.languageCode:
            options_dict["language"] = config_request.languageCode

        with self._model_lock:
            asr_model = self._model.get('asr')
            if asr_model is None:
                raise RuntimeError("ASR model not loaded")
            result = asr_model.transcribe(load_audio(audio.file), **options_dict)
            logger.debug(f"Transcribing result: {result}")

    def _monitor_idleness(self):
        if Config.ASR_MODEL_IDLE_TIMEOUT <= 0:
            return
        while True:
            time.sleep(15)
            if time.time() - self._last_activity_time > Config.ASR_MODEL_IDLE_TIMEOUT:
                with self._model_lock:
                    self._release_model()
                    break

    def _release_model(self):
        self._model = {}
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Model {Config.ASR_MODEL} unloaded due to timeout {Config.ASR_MODEL_IDLE_TIMEOUT}")

    def _ensure_model_loaded(self):
        """Ensures the model is loaded and updates last activity time."""
        self._last_activity_time = time.time()
        with self._model_lock:
            if not self._model.get('asr'):
                self.load_model()
