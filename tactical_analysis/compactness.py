from typing import List, Tuple
import numpy as np

PITCH_AREA = 105.0 * 68.0  # 7140 m^2


def compute_compactness(player_positions: List[Tuple[float, float]]) -> float:
    """
    Compactness score in [0, 1].
    1 = fully compact (all players in same spot)
    0 = spread across the entire pitch
    Uses convex hull area normalized by max pitch area.
    """
    if len(player_positions) < 3:
        return 1.0
    from scipy.spatial import ConvexHull
    pts = np.array(player_positions)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # 2D volume = area
    except Exception:
        hull_area = 0.0
    compactness = 1.0 - (hull_area / PITCH_AREA)
    return round(max(0.0, min(1.0, compactness)), 4)
