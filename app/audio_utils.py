import logging
import os
import tempfile
from typing import BinaryIO

import ffmpeg
import numpy as np

from app.config import Config

logger = logging.getLogger(__name__)


def load_audio(file: BinaryIO, sample_rate: int = Config.SAMPLE_RATE):
    logger.info(f"Loading audio file with sample_rate={sample_rate}")

    tmp_path = None
    try:
        # Write to a temporary file so ffmpeg can seek freely.
        # Container formats like MP4/MOV/MKV store metadata (moov atom) at the
        # end of the file, which requires seeking — impossible on a pipe.
        audio_data = file.read()
        audio_size_mb = len(audio_data) / (1024 * 1024)
        logger.debug(f"Read {audio_size_mb:.2f} MB of audio data from file")

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp.name
        tmp.write(audio_data)
        tmp.close()

        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        logger.debug("Starting ffmpeg processing: decoding, down-mixing to mono, and resampling")
        out, _ = (
            ffmpeg.input(tmp_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
        logger.debug(f"ffmpeg processing complete, output size: {len(out)} bytes")

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
        logger.error(f"ffmpeg failed to process audio: {error_msg}")
        raise RuntimeError(f"Failed to load audio: {error_msg}") from e

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Convert to numpy array and normalize
    audio_array = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    duration_seconds = len(audio_array) / sample_rate
    logger.info(f"Audio loaded successfully: {len(audio_array)} samples, {duration_seconds:.2f}s duration")

    return audio_array
