from typing import List, Tuple, Dict

CHANNEL_BOUNDS = {
    "left":   (0.0, 22.67),
    "center": (22.67, 45.33),
    "right":  (45.33, 68.0)
}


def compute_overload_ratio(
    attacking_positions: List[Tuple[float, float]],
    defending_positions: List[Tuple[float, float]],
    channel: str = "right"
) -> Dict:
    """
    Count players in the target channel and compute overload ratio.
    Player position is (x_world, y_world); channel is determined by y_world.
    """
    y_min, y_max = CHANNEL_BOUNDS.get(channel, (0, 68))
    atk = [p for p in attacking_positions if y_min <= p[1] < y_max]
    dfn = [p for p in defending_positions if y_min <= p[1] < y_max]
    ratio = len(atk) / max(len(dfn), 1)
    return {
        "channel": channel,
        "attacking_count": len(atk),
        "defending_count": len(dfn),
        "overload_ratio": round(ratio, 3)
    }


def detect_worst_overload(
    attacking_positions: List[Tuple[float, float]],
    defending_positions: List[Tuple[float, float]]
) -> Dict:
    """Find the channel with the highest overload ratio."""
    best = {"overload_ratio": 0.0, "channel": "center"}
    for ch in ["left", "center", "right"]:
        result = compute_overload_ratio(attacking_positions, defending_positions, ch)
        if result["overload_ratio"] > best["overload_ratio"]:
            best = result
    return best
