from pydantic import BaseModel
from typing import Optional, List


class LocalizationRecord(BaseModel):
    clip_id: str
    frame_id: int
    timestamp: float
    player_id: int
    x_image: float
    y_image: float
    x_world: Optional[float] = None
    y_world: Optional[float] = None
    bbox: List[int]       # [x, y, width, height]
    confidence: float
    source: str = "yolov8"   # "yolov8" or "synthetic"
