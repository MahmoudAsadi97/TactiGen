from pydantic import BaseModel
from typing import List, Dict, Optional


class EvidenceObject(BaseModel):
    clip_id: str
    time_window: str
    detected_pattern: str
    pattern_confidence: float
    supporting_metrics: Dict
    model_confidence: Dict
    supporting_frames: List[str]
    anticipated_action: Dict
    limitations: List[str]
    report_status: str = "ok"   # "ok" | "low_confidence_review"
