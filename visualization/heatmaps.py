"""Tactical heatmap generation."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple
from visualization.pitch_plots import draw_pitch


def generate_tactical_heatmap(
    player_positions: List[Tuple[float, float]],
    pitch_dims: Tuple[float, float] = (105, 68),
    title: str = "Tactical Heatmap",
    output_path: str = None
) -> str:
    if not player_positions:
        return ""
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, pitch_dims[0], pitch_dims[1])

    xs = [p[0] for p in player_positions]
    ys = [p[1] for p in player_positions]
    sns.kdeplot(x=xs, y=ys, ax=ax, cmap="Reds", fill=True, alpha=0.5, levels=10,
                clip=((0, pitch_dims[0]), (0, pitch_dims[1])))

    ax.set_title(title, fontsize=14, color="white", pad=10)
    fig.patch.set_facecolor("#1a1a2e")

    if output_path is None:
        Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
        output_path = f"outputs/visualizations/heatmap_{title.replace(' ', '_')}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path
