"""Player trajectory visualization."""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from visualization.pitch_plots import draw_pitch


def generate_trajectory_plot(
    player_tracks: Dict[int, List[Tuple[float, float]]],
    pitch_dims: Tuple[float, float] = (105, 68),
    title: str = "Player Trajectories",
    output_path: str = None
) -> str:
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, pitch_dims[0], pitch_dims[1])

    colors = cm.tab10(np.linspace(0, 1, max(len(player_tracks), 1)))
    for (pid, track), color in zip(player_tracks.items(), colors):
        if len(track) < 2:
            continue
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8, label=f"P{pid}")
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        ax.scatter([xs[0]], [ys[0]], color=color, s=60, zorder=5)

    ax.set_title(title, fontsize=14, color="white", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    fig.patch.set_facecolor("#1a1a2e")

    if output_path is None:
        Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
        output_path = f"outputs/visualizations/trajectories_{title.replace(' ', '_')}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path
