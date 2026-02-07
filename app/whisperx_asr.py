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


class _ModelDict(TypedDict, total=False):
    """Type definition for the model dictionary."""
    asr: Any  # whisperx.Model type not publicly exposed
    diarize: DiarizationPipeline
    alignment: dict[str, Any]


class WhisperxASR:
    _model: _ModelDict = {}

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

        # Initialize alignment model cache as empty dictionary
        self._model['alignment'] = {}

        Thread(target=self._monitor_idleness, daemon=True).start()

    def is_ready(self) -> bool:
        """Check if the ASR system is ready to handle transcription requests.

        Returns:
            True if core models (ASR and diarization) are loaded, False otherwise
        """
        with self._model_lock:
            return bool(self._model.get('asr')) and bool(self._model.get('diarize'))

    def transcribe(self, audio, config_request: ConfigRequest):
        logger.debug(f"Transcribing request: {config_request}")
        self._ensure_model_loaded()
        audio_file = load_audio(audio.file)
        result = self._perform_transcription(audio_file, config_request)

        if not config_request.features:
            return result

        if config_request.features.enableWordLevelTimestamps:
            result = self._perform_alignment(result, audio_file)

        if config_request.features.diarization and config_request.features.diarization.enableSpeakerDiarization:
            result = self._perform_diarization(result, audio_file, config_request.features.diarization)

        return result

    def _perform_transcription(self, audio, config_request: ConfigRequest):
        """Performs the actual transcription with the configured options.

        Args:
            audio: Audio file object with a .file attribute
            config_request: Configuration request containing language code and other options

        Returns:
            Transcription result dictionary from WhisperX
        """
        options_dict = {"batch_size": Config.ASR_BATCH_SIZE}
        if config_request.languageCode:
            options_dict["language"] = config_request.languageCode

        with self._model_lock:
            asr_model = self._model.get('asr')
            if asr_model is None:
                raise RuntimeError("ASR model not loaded")
            result = asr_model.transcribe(audio, **options_dict)
            logger.debug(f"Transcribing result: {result}")
            return result

    def _perform_alignment(self, transcription_result: dict, audio_file):
        """Performs word-level alignment on transcription segments.

        Loads and caches language-specific alignment models on demand. The alignment
        process adds precise timestamps for each word in the transcription.

        Args:
            transcription_result: Dictionary containing transcription segments and detected language
            audio_file: Loaded audio data (numpy array)

        Returns:
            Dictionary with aligned segments containing word-level timestamps
        """
        # Load alignment model for detected language (cached per language)
        alignment_cache = self._model.get('alignment', {})
        detected_language = transcription_result["language"]

        if detected_language not in alignment_cache:
            logger.info(f"Loading alignment model for language: {detected_language}")
            alignment_cache[detected_language] = whisperx.load_align_model(
                language_code=detected_language,
                device=Config.ASR_DEVICE,
            )

        alignment_model, metadata = alignment_cache[detected_language]

        result = whisperx.align(
            transcription_result["segments"],
            alignment_model,
            metadata,
            audio_file,
            Config.ASR_DEVICE,
            return_char_alignments=False
        )
        logger.debug(f"Alignment result: {result}")
        return result

    def _perform_diarization(self, transcription_result: dict, audio_file, diarization_config):
        """Performs speaker diarization to identify who spoke when.

        Uses pyannote.audio models to detect different speakers in the audio and assigns
        speaker labels to each word in the transcription.

        Args:
            transcription_result: Dictionary containing aligned transcription segments with words
            audio_file: Loaded audio data (numpy array)
            diarization_config: DiarizationConfig object with min/max speaker counts

        Returns:
            Dictionary with transcription segments annotated with speaker labels
        """
        diarize_model = self._model.get('diarize')
        if diarize_model is None:
            raise RuntimeError("Diarization model not loaded")

        diarize_segments = diarize_model(
            audio_file,
            min_speakers=diarization_config.minSpeakerCount,
            max_speakers=diarization_config.maxSpeakerCount
        )
        logger.debug(f"Diarization result: {diarize_segments}")

        result = whisperx.assign_word_speakers(diarize_segments, transcription_result)
        return result

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
