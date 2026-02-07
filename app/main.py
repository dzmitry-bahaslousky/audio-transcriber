from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import RedirectResponse

from app.log_config import get_logger
from app.model import ConfigRequest
from app.whisperx_asr import WhisperxASR

logger = get_logger(__name__)

whisperx_asr = WhisperxASR()
whisperx_asr.load_model()

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
    return whisperx_asr.transcribe(file, config)


@api.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


@api.get("/health/liveness", tags=["Health Check"])
async def liveness_check():
    return {"status": "alive"}


@api.get("/health/readiness", tags=["Health Check"])
async def readiness_check():
    if not whisperx_asr.is_ready():
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}
