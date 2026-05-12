import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from core.tumor_cell import CellState

STATE_COLORS = {
    CellState.GSC:       "#7F77DD",
    CellState.PROLIF:    "#D85A30",
    CellState.INVASIVE:  "#EF9F27",
    CellState.QUIESCENT: "#888780",
    CellState.NECROTIC:  "#2C2C2A",
}


def plot_spatial_snapshot(engine, title=""):
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


def plot_treatment_spatial(engine, title=""):
    """
    Month 5 — Three panel treatment spatial figure.
    Panel 1: TMZ concentration field + MGMT status overlay
    Panel 2: DNA damage map (per-voxel mean damage of cells there)
    Panel 3: GSC positions highlighted (recurrence seeds)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: TMZ field
    ax = axes[0]
    im = ax.imshow(engine.treatment.tmz, cmap="YlOrRd", origin="lower",
                   vmin=0, vmax=engine.treatment.p["tmz_vessel_conc"])
    plt.colorbar(im, ax=ax, label="TMZ (uM)")
    # Overlay MGMT status: green = methylated (sensitive), red = unmethylated
    for cell in engine.cells:
        if cell.state == CellState.NECROTIC:
            continue
        color = "#00CC44" if cell.mgmt_methylated else "#CC2200"
        ax.plot(cell.x, cell.y, ".", color=color, ms=1.5, alpha=0.5)
    ax.set_title(f"TMZ field + MGMT status  {title}")
    m_patch = mpatches.Patch(color="#00CC44", label="MGMT+ (sensitive)")
    u_patch = mpatches.Patch(color="#CC2200", label="MGMT- (resistant)")
    ax.legend(handles=[m_patch, u_patch], fontsize=7, loc="upper right")

    # Panel 2: DNA damage map
    ax = axes[1]
    H, W = engine.grid.oxygen.shape
    dmg_map = np.zeros((H, W))
    count_map = np.zeros((H, W))
    for cell in engine.cells:
        dmg_map[cell.y, cell.x]   += cell.dna_damage
        count_map[cell.y, cell.x] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_dmg = np.where(count_map > 0, dmg_map / count_map, 0)
    im2 = ax.imshow(avg_dmg, cmap="hot", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax, label="Mean DNA damage (0=none, 1=lethal)")
    ax.set_title(f"DNA damage map  {title}")

    # Panel 3: GSC positions (recurrence seeds)
    ax = axes[2]
    ax.imshow(engine.grid.oxygen, cmap="Greys", origin="lower",
              vmin=0, vmax=40, alpha=0.4)
    # Show all cells faintly
    for cell in engine.cells:
        if cell.state != CellState.GSC:
            color = STATE_COLORS.get(cell.state, "gray")
            ax.plot(cell.x, cell.y, ".", color=color, ms=1, alpha=0.2)
    # GSCs prominently
    gsc_cells = [c for c in engine.cells if c.state == CellState.GSC]
    for cell in gsc_cells:
        ax.plot(cell.x, cell.y, "*", color="#7F77DD", ms=8, alpha=0.9)
    ax.set_title(f"GSC positions (n={len(gsc_cells)}) — recurrence seeds  {title}")

    plt.tight_layout()
    return fig


def plot_treatment_response(history):
    """
    Month 5 — Four-panel treatment response figure.
    Panel 1: Tumor growth with treatment phase shading
    Panel 2: TMZ concentration over time
    Panel 3: MGMT methylated vs unmethylated cell counts
    Panel 4: RT fractions delivered + GSC count
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    times = [h["day"] for h in history]

    def shade_treatment(ax):
        """Shade concurrent treatment phase (days 1-42)."""
        ax.axvspan(1, 42, alpha=0.08, color="#CC3300", label="Concurrent RT+TMZ")

    # Panel 1: Tumor growth
    ax = axes[0]
    shade_treatment(ax)
    ax.plot(times, [h["n_cells"] for h in history],
            color="#D85A30", lw=2, label="Total cells")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Tumor cell count")
    ax.set_title("Tumor growth under treatment")
    ax.legend(fontsize=8)

    # Panel 2: TMZ concentration over time
    ax = axes[1]
    shade_treatment(ax)
    ax.plot(times, [h["tmz_mean"] for h in history],
            color="#E8A838", lw=2, label="TMZ mean")
    ax.plot(times, [h["tmz_max"] for h in history],
            color="#E8A838", lw=1, ls="--", label="TMZ max")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("TMZ concentration (uM)")
    ax.set_title("TMZ pharmacokinetics")
    ax.legend(fontsize=8)

    # Panel 3: MGMT methylated vs unmethylated
    ax = axes[2]
    shade_treatment(ax)
    ax.plot(times, [h["n_methylated"]   for h in history],
            color="#00AA44", lw=2, label="MGMT+ methylated (sensitive)")
    ax.plot(times, [h["n_unmethylated"] for h in history],
            color="#CC2200", lw=2, label="MGMT- unmethylated (resistant)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.set_title("MGMT methylation — treatment sensitivity")
    ax.legend(fontsize=8)

    # Panel 4: RT fractions + GSC count
    ax = axes[3]
    shade_treatment(ax)
    ax.plot(times, [h["rt_fractions"] for h in history],
            color="#3A6BCC", lw=2, label="RT fractions delivered")
    ax2 = ax.twinx()
    ax2.plot(times, [h["n_gsc"] for h in history],
             color="#7F77DD", lw=2, ls="--", label="GSC count")
    ax2.set_ylabel("GSC count", color="#7F77DD")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("RT fractions")
    ax.set_title("RT schedule + GSC survival")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="center right")

    plt.tight_layout()
    return fig


def plot_immune_snapshot(engine, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.imshow(engine.grid.oxygen, cmap="Greys", origin="lower",
              vmin=0, vmax=40, alpha=0.3)
    for tam in engine.tams:
        color = "#3A6BCC" if tam.state.name == "M1" else "#CC3A3A"
        ax.plot(tam.x, tam.y, "s", color=color, ms=4, alpha=0.8)
    for cell in engine.cells:
        if cell.state == CellState.PROLIF:
            ax.plot(cell.x, cell.y, ".", color="#D85A30", ms=1, alpha=0.2)
    ax.plot([], [], "s", color="#3A6BCC",
            label=f"M1 ({sum(1 for t in engine.tams if t.state.name=='M1')})")
    ax.plot([], [], "s", color="#CC3A3A",
            label=f"M2 ({sum(1 for t in engine.tams if t.state.name=='M2')})")
    ax.set_title(f"TAM polarisation  {title}")
    ax.legend(fontsize=8)

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
    ax.plot([], [], "^", color="#2DA84B",
            label=f"T active ({sum(1 for t in engine.tcells if t.state.name=='ACTIVE')})")
    ax.plot([], [], "^", color="#888888",
            label=f"T exhausted ({sum(1 for t in engine.tcells if t.state.name=='EXHAUSTED')})")
    ax.plot([], [], "D", color="#DAA520", label=f"MDSC ({len(engine.mdscs)})")
    ax.plot([], [], "v", color="#9B59B6", label=f"Treg ({len(engine.tregs)})")
    ax.set_title(f"Immune positions  {title}")
    ax.legend(fontsize=7)

    ax = axes[2]
    H, W = engine.signaling.tgf_beta.shape
    overlay = np.zeros((H, W, 3))
    tgf_n  = np.clip(engine.signaling.tgf_beta / max(engine.signaling.tgf_beta.max(), 1e-6), 0, 1)
    ifng_n = np.clip(engine.signaling.ifng     / max(engine.signaling.ifng.max(),     1e-6), 0, 1)
    overlay[:,:,0] = tgf_n
    overlay[:,:,1] = ifng_n
    ax.imshow(overlay, origin="lower")
    ax.set_title("TGF-\u03b2 (red) vs IFN-\u03b3 (green)")

    plt.tight_layout()
    return fig


def plot_growth_curve(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    times = [h["day"] for h in history]

    ax = axes[0]
    ax.axvspan(1, 42, alpha=0.08, color="#CC3300", label="RT+TMZ phase")
    ax.plot(times, [h["n_cells"] for h in history], color="#D85A30", lw=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total tumor cell count")
    ax.set_title("Tumor growth")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axvspan(1, 42, alpha=0.08, color="#CC3300")
    for state, color in STATE_COLORS.items():
        counts = [h["tumor_states"].get(state.name, 0) for h in history]
        ax.plot(times, counts, color=color, lw=1.5, label=state.name)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell count")
    ax.set_title("Cell state dynamics")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_immune_timeseries(history):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    times = [h["day"] for h in history]

    def shade(ax):
        ax.axvspan(1, 42, alpha=0.08, color="#CC3300")

    ax = axes[0]
    shade(ax)
    ax.plot(times, [h["n_tam_m1"] for h in history],
            color="#3A6BCC", lw=2, label="M1 (anti-tumor)")
    ax.plot(times, [h["n_tam_m2"] for h in history],
            color="#CC3A3A", lw=2, label="M2 (pro-tumor)")
    final_m1 = history[-1]["n_tam_m1"]
    final_m2 = history[-1]["n_tam_m2"]
    total = final_m1 + final_m2
    if total > 0:
        ax.text(0.05, 0.95, f"Final M2: {final_m2/total*100:.0f}%",
                transform=ax.transAxes, fontsize=8, color="#CC3A3A", va="top")
    ax.set_title("TAM Polarisation")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax = axes[1]
    shade(ax)
    ax.plot(times, [h["n_tcell_active"] for h in history],
            color="#2DA84B", lw=2, label="Active CD8+")
    ax.plot(times, [h["n_tcell_exh"] for h in history],
            color="#888888", lw=2, label="Exhausted CD8+")
    ax.set_title("T Cell Exhaustion")
    ax.set_xlabel("Time (days)")
    ax.legend(fontsize=8)

    ax = axes[2]
    shade(ax)
    ax.plot(times, [h["tgf_mean"]  for h in history],
            color="#7F77DD", lw=2, label="TGF-\u03b2")
    ax.plot(times, [h["il10_mean"] for h in history],
            color="#E8873A", lw=2, label="IL-10")
    ax.plot(times, [h["ifng_mean"] for h in history],
            color="#2DA84B", lw=2, label="IFN-\u03b3")
    ax.set_title("Cytokine Balance")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mean conc. (pg/mL)")
    ax.legend(fontsize=8)

    ax = axes[3]
    shade(ax)
    ax.plot(times, [h["n_mdsc"] for h in history],
            color="#DAA520", lw=2, label="MDSC")
    ax.plot(times, [h["n_treg"] for h in history],
            color="#9B59B6", lw=2, label="Treg")
    ax.set_title("MDSC and Treg")
    ax.set_xlabel("Time (days)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig