"""Standard football pitch drawing helper."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_pitch(ax, pitch_length: float = 105.0, pitch_width: float = 68.0, color: str = "#4a7c59"):
    ax.set_facecolor(color)
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    lw, lc = 2, "white"
    # Outer boundary
    ax.plot([0, pitch_length, pitch_length, 0, 0],
            [0, 0, pitch_width, pitch_width, 0], color=lc, linewidth=lw)
    # Halfway line
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color=lc, linewidth=lw)
    # Center circle
    center = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color=lc, fill=False, linewidth=lw)
    ax.add_patch(center)
    # Penalty areas
    for x0, x1 in [(0, 16.5), (pitch_length - 16.5, pitch_length)]:
        rect = patches.Rectangle((x0, (pitch_width - 40.32) / 2), abs(x1 - x0), 40.32,
                                   linewidth=lw, edgecolor=lc, facecolor="none")
        ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.axis("off")
