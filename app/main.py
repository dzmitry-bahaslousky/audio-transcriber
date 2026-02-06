from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import RedirectResponse

from app.log_config import get_logger
from app.model import ConfigRequest
from app.whisperx_asr import WhisperxASR

logger = get_logger(__name__)

whisperx = WhisperxASR()
whisperx.load_model()

api = FastAPI()

@api.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@api.post("/transcribe", tags=["Transcribe"])
async def transcribe(
        file: UploadFile = File(...),
        config: ConfigRequest = Form(...),
):
    logger.info(f"Transcribe request received: {config}")
    whisperx.transcribe(file, config)
    return {"transcription": "transcription"}