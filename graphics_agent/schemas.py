
from pydantic import BaseModel
from typing import Optional, Dict, Literal

class TextToGraphicsRequest(BaseModel):
    text: str
    chart_type: str = "infographic"
    data: Optional[Dict] = None
    tone: str = "neutral"
    color_scheme: str = "default"
    subject: str = "general"
    style_preset: Literal["photorealistic", "anime", "impressionism", "digital-art", "comic-book"] = "digital-art"
    negative_prompt: Optional[str] = None
    width: int = 768
    height: int = 768
    enhance_image: bool = False

class TextToGraphicsResponse(BaseModel):
    graphics_file: str
    message: str
