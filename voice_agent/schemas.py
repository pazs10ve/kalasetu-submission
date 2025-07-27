
from pydantic import BaseModel
from typing import Literal

class TextToAudioRequest(BaseModel):
    text: str
    language: str = "en"
    accent: str = "us"
    ambience: Literal["none", "cafe", "news_studio", "nature"] = "none"
    creativity: float = 0.7  # Corresponds to fine_temperature
    stability: float = 0.3  # Corresponds to coarse_temperature

class TextToAudioResponse(BaseModel):
    audio_file: str
    message: str

