from pydantic import BaseModel

class DiarizationConfig(BaseModel):
    minSpeakerCount: int
    maxSpeakerCount: int


class FeaturesConfig(BaseModel):
    enableWordLevelTimestamps: bool
    diarization: DiarizationConfig


class Config(BaseModel):
    language: str
    features: FeaturesConfig


class TranscribeRequest(BaseModel):
    config: Config