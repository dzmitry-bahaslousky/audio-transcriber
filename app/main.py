from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse

from app.model import TranscribeRequest

app = FastAPI()

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/transcribe")
async def transcribe(
        file: UploadFile = File(...),
        config: TranscribeRequest = Body(...),
):
    return {"transcription": "transcription"}