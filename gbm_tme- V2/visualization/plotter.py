import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from core.tumor_cell import CellState

STATE_COLORS = {
    CellState.GSC:       "#7F77DD",
    CellState.PROLIF:    "#D85A30",
    CellState.INVASIVE:  "#EF9F27",
    CellState.QUIESCENT: "#888780",
    CellState.NECROTIC:  "#2C2C2A",
}


def plot_spatial_snapshot(engine, title=""):
    """O2 field + cells, cell states, glucose."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    im = ax.imshow(engine.grid.oxygen, cmap="Blues_r", origin="lower",
                   vmin=0, vmax=40)
    plt.colorbar(im, ax=ax, label="pO2 (mmHg)")
    for cell in engine.cells:
        color = STATE_COLORS.get(cell.state, "white")
        ax.plot(cell.x, cell.y, "o", color=color, ms=2, alpha=0.7)
    vy, vx = np.where(engine.grid.vessels)
    ax.plot(vx, vy, "+", color="red", ms=4, alpha=0.5)
    ax.set_title(f"O2 field + cells  {title}")
    ax.set_xlabel("x (voxels)")

    ax = axes[1]
    state_img = np.zeros((*engine.grid.oxygen.shape, 3))
    for cell in engine.cells:
        rgb = mcolors.to_rgb(STATE_COLORS.get(cell.state, "gray"))
        state_img[cell.y, cell.x] = rgb
    ax.imshow(state_img, origin="lower")
    ax.set_title("Tumor cell states")
    for state, color in STATE_COLORS.items():
        ax.plot([], [], "o", color=color, label=state.name)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[2]
    im2 = ax.imshow(engine.grid.glucose, cmap="Greens_r", origin="lower",
                    vmin=0, vmax=5)
    plt.colorbar(im2, ax=ax, label="Glucose (mM)")
    ax.set_title("Glucose field")

    plt.tight_layout()
    return fig


def plot_immune_snapshot(engine, title=""):
    """
    Month 4 — Three-panel immune spatial map.
    Panel 1: TAM M1 (blue dots) vs M2 (red dots) positions
    Panel 2: T cell positions — active (green) vs exhausted (grey)
    Panel 3: IFN-g vs TGF-b field overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: TAM polarisation map
    ax = axes[0]
    ax.imshow(engine.grid.oxygen, cmap="Greys", origin="lower",
              vmin=0, vmax=40, alpha=0.3)
    for tam in engine.tams:
        color = "#3A6BCC" if tam.state.name == "M1" else "#CC3A3A"
        ax.plot(tam.x, tam.y, "s", color=color, ms=4, alpha=0.8)
    ax.plot([], [], "s", color="#3A6BCC", label=f"M1 TAM ({sum(1 for t in engine.tams if t.state.name=='M1')})")
    ax.plot([], [], "s", color="#CC3A3A", label=f"M2 TAM ({sum(1 for t in engine.tams if t.state.name=='M2')})")
    # Also show tumor boundary for context
    for cell in engine.cells:
        if cell.state == CellState.PROLIF:
            ax.plot(cell.x, cell.y, ".", color="#D85A30", ms=1, alpha=0.2)
    ax.set_title(f"TAM polarisation  {title}")
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: T cell states
    ax = axes[1]
    ax.imshow(engine.grid.oxygen, cmap="Greys", origin="lower",
              vmin=0, vmax=40, alpha=0.3)
    for tc in engine.tcells:
        color = "#2DA84B" if tc.state.name == "ACTIVE" else "#888888"
        ax.plot(tc.x, tc.y, "^", color=color, ms=4, alpha=0.8)
    for mdsc in engine.mdscs:
        ax.plot(mdsc.x, mdsc.y, "D", color="#DAA520", ms=3, alpha=0.7)
    for treg in engine.tregs:
        ax.plot(treg.x, treg.y, "v", color="#9B59B6", ms=3, alpha=0.7)
    ax.plot([], [], "^", color="#2DA84B", label=f"T active ({sum(1 for t in engine.tcells if t.state.name=='ACTIVE')})")
    ax.plot([], [], "^", color="#888888", label=f"T exhausted ({sum(1 for t in engine.tcells if t.state.name=='EXHAUSTED')})")
    ax.plot([], [], "D", color="#DAA520", label=f"MDSC ({len(engine.mdscs)})")
    ax.plot([], [], "v", color="#9B59B6", label=f"Treg ({len(engine.tregs)})")
    ax.set_title(f"Immune cell positions  {title}")
    ax.legend(fontsize=7, loc="upper right")

    # Panel 3: IFN-g vs TGF-b overlay
    ax = axes[2]
    # Red = TGF-b (immunosuppressive), Green = IFN-g (anti-tumor)
    H, W = engine.signaling.tgf_beta.shape
    overlay = np.zeros((H, W, 3))
    tgf_norm = np.clip(engine.signaling.tgf_beta / max(engine.signaling.tgf_beta.max(), 1e-6), 0, 1)
    ifng_norm= np.clip(engine.signaling.ifng     / max(engine.signaling.ifng.max(),     1e-6), 0, 1)
    overlay[:,:,0] = tgf_norm   # red channel = TGF-b
    overlay[:,:,1] = ifng_norm  # green channel = IFN-g
    ax.imshow(overlay, origin="lower")
    ax.set_title("TGF-\u03b2 (red) vs IFN-\u03b3 (green)")
    ax.text(5, 5, "Red=immunosuppressed\nGreen=anti-tumor", fontsize=7,
            color="white", va="bottom")

    plt.tight_layout()
    return fig


def plot_signaling_snapshot(engine, title=""):
    """VEGF, TGF-b, IL-10, IFN-g spatial fields."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    panels = [
        (engine.signaling.vegf,     "Reds",    "VEGF (pg/mL)",     "VEGF"),
        (engine.signaling.tgf_beta, "Purples", "TGF-\u03b2 (pg/mL)", "TGF-\u03b2"),
        (engine.signaling.il10,     "Oranges", "IL-10 (pg/mL)",    "IL-10"),
        (engine.signaling.ifng,     "Greens",  "IFN-\u03b3 (pg/mL)", "IFN-\u03b3"),
    ]

    for ax, (field, cmap, label, ttl) in zip(axes, panels):
        im = ax.imshow(field, cmap=cmap, origin="lower", vmin=0)
        plt.colorbar(im, ax=ax, label=label)
        vy, vx = np.where(engine.grid.vessels)
        ax.plot(vx, vy, "+", color="cyan", ms=3, alpha=0.3)
        ax.set_title(f"{ttl}  {title}")

    plt.tight_layout()
    return fig


def plot_growth_curve(history):
    """Tumor growth + cell state dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    times = [h["time_hr"] / 24 for h in history]

    ax = axes[0]
    ax.plot(times, [h["n_cells"] for h in history], color="#D85A30", lw=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total tumor cell count")
    ax.set_title("Tumor growth")

    ax = axes[1]
    for state, color in STATE_COLORS.items():
        counts = [h["tumor_states"].get(state.name, 0) for h in history]
        ax.plot(times, counts, color=color, lw=1.5, label=state.name)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.set_title("Tumor cell state dynamics")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_immune_timeseries(history):
    """
    Month 4 — Four-panel immune time series.
    Panel 1: TAM M1 vs M2 counts over time
    Panel 2: T cell active vs exhausted over time
    Panel 3: IFN-g mean vs TGF-b mean (immunosuppression balance)
    Panel 4: MDSC and Treg counts
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    times = [h["time_hr"] / 24 for h in history]

    # TAM polarisation
    ax = axes[0]
    ax.plot(times, [h["n_tam_m1"] for h in history],
            color="#3A6BCC", lw=2, label="M1 TAM (anti-tumor)")
    ax.plot(times, [h["n_tam_m2"] for h in history],
            color="#CC3A3A", lw=2, label="M2 TAM (pro-tumor)")
    ax.set_title("TAM Polarisation")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.legend(fontsize=8)
    # Annotate: GBM should show M2 dominance
    final_m1 = history[-1]["n_tam_m1"]
    final_m2 = history[-1]["n_tam_m2"]
    total_tam = final_m1 + final_m2
    if total_tam > 0:
        m2_frac = final_m2 / total_tam * 100
        ax.text(0.05, 0.95, f"Final M2 fraction: {m2_frac:.0f}%",
                transform=ax.transAxes, fontsize=8,
                color="#CC3A3A", va="top")

    # T cell exhaustion
    ax = axes[1]
    ax.plot(times, [h["n_tcell_active"] for h in history],
            color="#2DA84B", lw=2, label="Active CD8+")
    ax.plot(times, [h["n_tcell_exh"] for h in history],
            color="#888888", lw=2, label="Exhausted CD8+")
    ax.set_title("T Cell Exhaustion")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.legend(fontsize=8)

    # Cytokine balance — the immunosuppression landscape
    ax = axes[2]
    ax.plot(times, [h["tgf_mean"] for h in history],
            color="#7F77DD", lw=2, label="TGF-\u03b2 mean")
    ax.plot(times, [h["il10_mean"] for h in history],
            color="#E8873A", lw=2, label="IL-10 mean")
    ax.plot(times, [h["ifng_mean"] for h in history],
            color="#2DA84B", lw=2, label="IFN-\u03b3 mean")
    ax.set_title("Cytokine Balance")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mean conc. (pg/mL)")
    ax.legend(fontsize=8)

    # MDSC and Treg
    ax = axes[3]
    ax.plot(times, [h["n_mdsc"] for h in history],
            color="#DAA520", lw=2, label="MDSC")
    ax.plot(times, [h["n_treg"] for h in history],
            color="#9B59B6", lw=2, label="Treg")
    ax.set_title("MDSC and Treg")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_signaling_timeseries(history):
    """VEGF + angiogenesis, TGF-b, IL-10 time series."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    times = [h["time_hr"] / 24 for h in history]

    ax = axes[0]
    ax.plot(times, [h["vegf_mean"] for h in history],
            color="#C93535", lw=2, label="VEGF mean")
    ax.plot(times, [h["vegf_max"]  for h in history],
            color="#C93535", lw=1, ls="--", label="VEGF max")
    ax2 = ax.twinx()
    ax2.plot(times, [h["vessel_count"] for h in history],
             color="#E8A838", lw=1.5, ls=":", label="Vessel count")
    ax2.set_ylabel("Vessel count", color="#E8A838")
    ax.set_title("VEGF + Angiogenesis")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("VEGF (pg/mL)")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="center right")

    ax = axes[1]
    ax.plot(times, [h["tgf_mean"] for h in history], color="#7F77DD", lw=2)
    ax.set_title("TGF-\u03b2 (immunosuppression)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("TGF-\u03b2 (pg/mL)")

    ax = axes[2]
    ax.plot(times, [h["il10_mean"] for h in history], color="#E8873A", lw=2)
    ax.set_title("IL-10 (immunosuppression)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("IL-10 (pg/mL)")

    plt.tight_layout()
    return fig