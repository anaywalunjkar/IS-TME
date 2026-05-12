import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.engine import SimulationEngine
from visualization.plotter import (
    plot_spatial_snapshot,
    plot_treatment_spatial,
    plot_treatment_response,
    plot_immune_snapshot,
    plot_growth_curve,
    plot_immune_timeseries,
)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from config.params import TREATMENT

    print("=" * 55)
    print("GBM TME Simulation — Month 5: Treatment Module")
    print("=" * 55)
    print(f"MGMT scenario:     {TREATMENT['mgmt_scenario']}")
    print(f"Treatment active:  {TREATMENT['treatment_active']}")
    print(f"Simulation length: 42 days (Stupp concurrent phase)")
    print("=" * 55)

    sim = SimulationEngine()

    print("\nRunning 42-day simulation...")
    history = sim.run()

    # ── Terminal summary ──────────────────────────────────────────
    last = history[-1]
    print(f"\n{'='*55}")
    print(f"FINAL STATE — Day {last['day']:.0f}")
    print(f"{'='*55}")

    print(f"\nTumor:")
    print(f"  Total cells:      {last['n_cells']}")
    for state, count in last["tumor_states"].items():
        print(f"  {state:12s}:   {count}")

    print(f"\nMGMT status (live cells):")
    print(f"  Methylated (sensitive):   {last['n_methylated']}")
    print(f"  Unmethylated (resistant): {last['n_unmethylated']}")
    print(f"  GSC (always resistant):   {last['n_gsc']}")

    print(f"\nTreatment delivered:")
    print(f"  TMZ active at end:  {last['tmz_active']}")
    print(f"  RT fractions done:  {last['rt_fractions']} / 30")
    print(f"  TMZ max conc:       {last['tmz_max']:.3f} uM")

    print(f"\nImmune compartment:")
    m1 = last["n_tam_m1"]
    m2 = last["n_tam_m2"]
    total_tam = m1 + m2
    m2_frac = m2 / total_tam * 100 if total_tam > 0 else 0
    print(f"  TAM M1:          {m1}")
    print(f"  TAM M2:          {m2}  ({m2_frac:.0f}% M2)")
    print(f"  T active:        {last['n_tcell_active']}")
    print(f"  T exhausted:     {last['n_tcell_exh']}")
    print(f"  MDSC:            {last['n_mdsc']}")
    print(f"  Treg:            {last['n_treg']}")

    print(f"\nSignaling (day 42):")
    print(f"  TGF-b mean:  {last['tgf_mean']:.4f} pg/mL")
    print(f"  IL-10 mean:  {last['il10_mean']:.4f} pg/mL")
    print(f"  IFN-g mean:  {last['ifng_mean']:.4f} pg/mL")
    print(f"{'='*55}")

    # ── Validation checks ─────────────────────────────────────────
    print("\nValidation checks:")

    # Check 1: methylated cells die faster
    if last["n_methylated"] < last["n_unmethylated"]:
        print("  PASS — Methylated cells < unmethylated (TMZ sensitivity confirmed)")
    else:
        print("  NOTE — Methylated >= unmethylated (may need more treatment time)")

    # Check 2: GSCs survive
    if last["n_gsc"] > 0:
        print(f"  PASS — GSCs survived treatment (n={last['n_gsc']}) — recurrence seeds present")
    else:
        print("  NOTE — No GSCs at day 42 (check gsc_tmz_resistance parameter)")

    # Check 3: RT fractions
    if last["rt_fractions"] == 30:
        print("  PASS — All 30 RT fractions delivered correctly")
    else:
        print(f"  NOTE — {last['rt_fractions']} RT fractions delivered (expected 30)")

    # Check 4: M2 dominance
    if m2_frac >= 70:
        print(f"  PASS — M2 TAM fraction {m2_frac:.0f}% >= 70% (cold tumor confirmed)")
    else:
        print(f"  NOTE — M2 fraction {m2_frac:.0f}% (below 70% target)")

    # ── Figures ───────────────────────────────────────────────────
    print("\nSaving figures...")

    fig1 = plot_spatial_snapshot(sim, "(day 42)")
    fig1.savefig("output_spatial.png", dpi=150)

    fig2 = plot_treatment_spatial(sim, "(day 42)")
    fig2.savefig("output_treatment_spatial.png", dpi=150)

    fig3 = plot_treatment_response(history)
    fig3.savefig("output_treatment_response.png", dpi=150)

    fig4 = plot_immune_snapshot(sim, "(day 42)")
    fig4.savefig("output_immune.png", dpi=150)

    fig5 = plot_growth_curve(history)
    fig5.savefig("output_growth.png", dpi=150)

    fig6 = plot_immune_timeseries(history)
    fig6.savefig("output_immune_timeseries.png", dpi=150)

    print("Saved 6 figures:")
    print("  output_spatial.png")
    print("  output_treatment_spatial.png  <- TMZ field + MGMT + DNA damage + GSCs")
    print("  output_treatment_response.png <- growth, TMZ PK, MGMT counts, RT schedule")
    print("  output_immune.png")
    print("  output_growth.png")
    print("  output_immune_timeseries.png")

    plt.show()