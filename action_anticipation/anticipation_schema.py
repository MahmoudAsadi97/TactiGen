from pydantic import BaseModel

ACTION_CLASSES = ["pass", "shot", "cross", "dribble", "duel",
                  "clearance", "interception", "foul", "ball_progression"]


class AnticipationResult(BaseModel):
    clip_id: str
    timestamp: float
    predicted_action: str
    confidence: float
    anticipation_window_seconds: int = 5
    model_used: str  # "transformer" or "heuristic"
    class_probabilities: dict = {}
