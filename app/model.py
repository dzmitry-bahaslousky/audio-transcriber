import json

from pydantic import BaseModel, model_validator


class DiarizationConfig(BaseModel):
    enableSpeakerDiarization: bool
    minSpeakerCount: int
    maxSpeakerCount: int


class FeaturesConfig(BaseModel):
    enableWordLevelTimestamps: bool
    diarization: DiarizationConfig


class Config(BaseModel):
    languageCode: str
    features: FeaturesConfig

    @model_validator(mode="before")
    @classmethod
    def parse_json_string(cls, data):
        if isinstance(data, str):
            return json.loads(data)
        return data