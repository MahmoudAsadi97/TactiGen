from typing import List, Tuple


def compute_team_width(player_positions: List[Tuple[float, float]]) -> float:
    """
    Horizontal spread of players in meters.
    team_width = max(x_world) - min(x_world)
    """
    if len(player_positions) < 2:
        return 0.0
    xs = [p[0] for p in player_positions]
    return round(max(xs) - min(xs), 2)
