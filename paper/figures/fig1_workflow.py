"""Figure 1: HyPlan workflow overview diagram.

Generates a pipeline diagram showing the five-stage workflow with
side inputs grouped into 'How to fly' and 'When to fly' categories.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-2.8, 3.8)
ax.set_aspect("equal")
ax.axis("off")

# ── Colors ──
C_STAGE = "#2c3e50"       # dark blue-gray for main pipeline boxes
C_STAGE_TXT = "white"
C_HOW = "#e67e22"         # orange for "how to fly"
C_WHEN = "#2980b9"        # blue for "when to fly"
C_OUTPUT = "#27ae60"      # green for outputs
C_ARROW = "#7f8c8d"       # gray arrows
C_BG_HOW = "#fef3e2"      # light orange background
C_BG_WHEN = "#e8f4fd"     # light blue background


def stage_box(x, y, label, color=C_STAGE, text_color=C_STAGE_TXT, w=1.7, h=0.7):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08", facecolor=color,
        edgecolor="none", zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=text_color, zorder=4)


def side_label(x, y, label, color, fontsize=6.5):
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=color, fontstyle="italic", zorder=4)


def arrow(x1, y1, x2, y2, color=C_ARROW):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
        zorder=2,
    )


def side_arrow(x1, y1, x2, y2, color):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                        linestyle="--"),
        zorder=2,
    )


def iter_arrow(x1, y1, x2, y2, color, rad=0.3):
    """Curved double-headed arrow to indicate iteration."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.9,
                        linestyle="--",
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )


# ── Main pipeline ──
xs = [0.8, 3.0, 5.2, 7.4, 9.6]
labels = [
    "Study\nArea",
    "Flight\nLines",
    "Swath\nAnalysis",
    "Mission\nPlan",
    "Outputs",
]
colors = [C_STAGE, C_STAGE, C_STAGE, C_STAGE, C_OUTPUT]
y_main = 1.0

for i, (x, lab, c) in enumerate(zip(xs, labels, colors)):
    stage_box(x, y_main, lab, color=c)
    if i < len(xs) - 1:
        arrow(x + 0.9, y_main, xs[i + 1] - 0.9, y_main)

# ── Output sub-labels ──
outputs = ["KML / GPX", "ForeFlight CSV", "Excel / ICARTT", "Folium maps"]
for i, txt in enumerate(outputs):
    ax.text(9.6, y_main - 0.55 - i * 0.32, txt, ha="center", va="top",
            fontsize=5.5, color="#1a7a3a", zorder=4)

# ── "How to fly" inputs (orange, top) ──
how_items = [
    (3.0, "Sensor"),
    (5.2, "Terrain / DEM"),
    (7.4, "Aircraft"),
    (7.4, "Winds"),
]
how_y = 2.8
# Background band
bg_how = FancyBboxPatch(
    (1.5, how_y - 0.45), 7.4, 0.9,
    boxstyle="round,pad=0.1", facecolor=C_BG_HOW,
    edgecolor=C_HOW, linewidth=0.8, linestyle="--", zorder=1,
)
ax.add_patch(bg_how)
ax.text(1.7, how_y + 0.35, "How to fly", fontsize=7, fontweight="bold",
        color=C_HOW, va="center", zorder=4)

how_positions = [(3.0, how_y), (5.2, how_y + 0.15), (6.8, how_y + 0.15), (8.0, how_y - 0.15)]
how_labels = ["Sensor", "Terrain / DEM", "Aircraft", "Winds"]
for (hx, hy), lab in zip(how_positions, how_labels):
    side_label(hx, hy, lab, C_HOW)
    # Find the closest pipeline stage
    target_x = min(xs, key=lambda sx: abs(sx - hx))
    side_arrow(hx, hy - 0.35, target_x, y_main + 0.4, C_HOW)

# ── "When to fly" inputs (blue, bottom) ──
when_y = -0.8
bg_when = FancyBboxPatch(
    (-0.3, when_y - 0.45), 9.2, 0.9,
    boxstyle="round,pad=0.1", facecolor=C_BG_WHEN,
    edgecolor=C_WHEN, linewidth=0.8, linestyle="--", zorder=1,
)
ax.add_patch(bg_when)
ax.text(-0.05, when_y + 0.35, "When to fly", fontsize=7, fontweight="bold",
        color=C_WHEN, va="center", zorder=4)

# Phenology and cloud climatology inform when to schedule the campaign
# based on the study area; cloud climatology additionally iterates with
# the mission plan (campaign duration depends on clear-sky probability
# and per-sortie flight time).
side_label(1.1, when_y + 0.15, "Vegetation\nphenology", C_WHEN)
side_arrow(1.1, when_y + 0.5, xs[0] + 0.15, y_main - 0.4, C_WHEN)

side_label(2.6, when_y + 0.15, "Cloud\nclimatology", C_WHEN)
# Cloud → Study Area (scheduling input)
side_arrow(2.6, when_y + 0.5, xs[0] + 0.5, y_main - 0.4, C_WHEN)
# Cloud ↔ Mission Plan (iterative: campaign duration estimate)
iter_arrow(2.9, when_y + 0.5, xs[3] - 0.3, y_main - 0.4, C_WHEN, rad=-0.25)

side_label(5.8, when_y + 0.15, "Solar\ngeometry", C_WHEN)
side_arrow(5.8, when_y + 0.5, xs[3] - 0.4, y_main - 0.4, C_WHEN)

# Cloud forecasts → Mission Plan (near-real-time, day-of-flight)
side_label(7.6, when_y + 0.15, "Cloud\nforecasts", C_WHEN)
side_arrow(7.6, when_y + 0.5, xs[3] + 0.1, y_main - 0.4, C_WHEN)

fig.tight_layout(pad=0.5)
fig.savefig("paper/figures/fig1_workflow.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("paper/figures/fig1_workflow.pdf", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Figure 1 saved.")
