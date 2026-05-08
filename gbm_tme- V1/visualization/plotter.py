import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from core.tumor_cell import CellState

STATE_COLORS = {
    CellState.GSC:       "#7F77DD",  # purple
    CellState.PROLIF:    "#D85A30",  # coral/red — active
    CellState.INVASIVE:  "#EF9F27",  # amber — warning
    CellState.QUIESCENT: "#888780",  # gray
    CellState.NECROTIC:  "#2C2C2A",  # near-black
}

def plot_spatial_snapshot(engine, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: O₂ field + cell positions
    ax = axes[0]
    im = ax.imshow(engine.grid.oxygen, cmap="Blues_r", origin="lower",
                   vmin=0, vmax=40)
    plt.colorbar(im, ax=ax, label="pO₂ (mmHg)")
    for cell in engine.cells:
        color = STATE_COLORS.get(cell.state, "white")
        ax.plot(cell.x, cell.y, "o", color=color, ms=2, alpha=0.7)
    # Mark vessels
    vy, vx = np.where(engine.grid.vessels)
    ax.plot(vx, vy, "+", color="red", ms=4, alpha=0.5)
    ax.set_title(f"O₂ field + cells  {title}")
    ax.set_xlabel("x (voxels)")

    # Panel 2: Cell state map
    ax = axes[1]
    state_img = np.zeros((*engine.grid.oxygen.shape, 3))
    for cell in engine.cells:
        rgb = mcolors.to_rgb(STATE_COLORS.get(cell.state, "gray"))
        state_img[cell.y, cell.x] = rgb
    ax.imshow(state_img, origin="lower")
    ax.set_title("Cell states")
    # Legend
    for state, color in STATE_COLORS.items():
        ax.plot([], [], "o", color=color, label=state.name)
    ax.legend(fontsize=7, loc="upper right")

    # Panel 3: Glucose
    ax = axes[2]
    im2 = ax.imshow(engine.grid.glucose, cmap="Greens_r", origin="lower",
                    vmin=0, vmax=5)
    plt.colorbar(im2, ax=ax, label="Glucose (mM)")
    ax.set_title("Glucose field")

    plt.tight_layout()
    return fig

def plot_growth_curve(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    times = [h["time_hr"] / 24 for h in history]  # convert to days

    # Total cell count
    ax = axes[0]
    ax.plot(times, [h["n_cells"] for h in history], color="#D85A30", lw=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total cell count")
    ax.set_title("Tumor growth")

    # State breakdown
    ax = axes[1]
    states = list(STATE_COLORS.keys())
    for state in states:
        counts = [h["states"].get(state.name, 0) for h in history]
        ax.plot(times, counts,
                color=STATE_COLORS[state], lw=1.5, label=state.name)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.set_title("Cell state dynamics")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig