import json
from typing import Optional

from pydantic import BaseModel, model_validator


class DiarizationConfig(BaseModel):
    enableSpeakerDiarization: bool = True
    minSpeakerCount: Optional[int] = None
    maxSpeakerCount: Optional[int] = None


class FeaturesConfig(BaseModel):
    enableWordLevelTimestamps: bool = False
    diarization: Optional[DiarizationConfig] = None


class ConfigRequest(BaseModel):
    languageCode: Optional[str] = None
    features: Optional[FeaturesConfig] = None

    @model_validator(mode="before")
    @classmethod
    def parse_json_string(cls, data):
        if isinstance(data, str):
            return json.loads(data)
        return data