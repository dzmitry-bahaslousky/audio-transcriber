import gc
import time
from threading import Thread, Lock

import torch

import whisperx
from whisperx.diarize import DiarizationPipeline

from app.log_config import get_logger
from app.config import Config

logger = get_logger(__name__)


class WhisperxASR:
    model = {
        'asr': None,
        'diarize': None,
        'alignment': None
    }

    model_lock = Lock()
    last_activity_time = time.time()

    def load_model(self):
        logger.info(f"Loading ASR model: '{Config.ASR_MODEL}' with device: '{Config.ASR_DEVICE}'")
        self.model['asr'] = whisperx.load_model(
            Config.ASR_MODEL,
            device=Config.ASR_DEVICE,
            compute_type=Config.ASR_COMPUTE_TYPE,
            download_root=Config.ASR_MODEL_CACHE_PATH
        )
        logger.info("Loaded ASR model")

        logger.info("Loading diarization model")
        self.model['diarize'] = DiarizationPipeline(
            use_auth_token=Config.HF_TOKEN,
            device=Config.ASR_DEVICE
        )

        Thread(target=self.monitor_idleness, daemon=True).start()

    def monitor_idleness(self):
        if Config.ASR_MODEL_IDLE_TIMEOUT <= 0:
            return
        while True:
            time.sleep(15)
            if time.time() - self.last_activity_time > Config.ASR_MODEL_IDLE_TIMEOUT:
                with self.model_lock:
                    self.release_model()
                    break

    def release_model(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        print("Model unloaded due to timeout")
