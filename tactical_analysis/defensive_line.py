from typing import List, Tuple
import numpy as np


def compute_defensive_line_height(
    defending_positions: List[Tuple[float, float]],
    attacking_direction: str = "right"
) -> float:
    """
    Median x_world of the 4 deepest defenders (furthest from opponent goal).
    attacking_direction='right' means defenders are closest to x=0.
    """
    if not defending_positions:
        return 0.0
    xs = sorted([p[0] for p in defending_positions])
    deepest = xs[:4] if attacking_direction == "right" else xs[-4:]
    return round(float(np.median(deepest)), 2)
