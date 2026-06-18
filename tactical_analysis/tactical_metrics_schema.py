from pydantic import BaseModel


class TacticalMetrics(BaseModel):
    clip_id: str
    time_window_start: float
    time_window_end: float
    team_width_meters: float
    compactness_score: float
    defensive_line_height_meters: float
    overload_channel: str
    attacking_count: int
    defending_count: int
    overload_ratio: float
