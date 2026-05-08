import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.engine import SimulationEngine
from visualization.plotter import (
    plot_spatial_snapshot,
    plot_immune_snapshot,
    plot_signaling_snapshot,
    plot_growth_curve,
    plot_immune_timeseries,
    plot_signaling_timeseries,
)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Initializing GBM TME simulation (Month 4 — immune compartment)...")
    sim = SimulationEngine()

    print("Running 30-day simulation...")
    history = sim.run()

    # Terminal summary
    print(f"\n{'='*50}")
    print(f"Final tumor cell count:  {len(sim.cells)}")
    print(f"Final vessel count:      {sim.grid.vessels.sum()}")
    print(f"\nImmune compartment (day 30):")
    m1 = sum(1 for t in sim.tams if t.state.name == "M1")
    m2 = sum(1 for t in sim.tams if t.state.name == "M2")
    ta = sum(1 for t in sim.tcells if t.state.name == "ACTIVE")
    te = sum(1 for t in sim.tcells if t.state.name == "EXHAUSTED")
    print(f"  TAM M1 (anti-tumor):   {m1}")
    print(f"  TAM M2 (pro-tumor):    {m2}")
    if m1 + m2 > 0:
        print(f"  M2 fraction:           {m2/(m1+m2)*100:.1f}%  (>70% = cold tumor validated)")
    print(f"  T cells active:        {ta}")
    print(f"  T cells exhausted:     {te}")
    print(f"  MDSCs:                 {len(sim.mdscs)}")
    print(f"  Tregs:                 {len(sim.tregs)}")
    print(f"\nSignaling (day 30):")
    print(f"  VEGF max:   {sim.signaling.vegf.max():.2f} pg/mL")
    print(f"  TGF-b max:  {sim.signaling.tgf_beta.max():.2f} pg/mL")
    print(f"  IL-10 max:  {sim.signaling.il10.max():.2f} pg/mL")
    print(f"  IFN-g max:  {sim.signaling.ifng.max():.2f} pg/mL")
    print(f"{'='*50}")

    # Figures
    fig1 = plot_spatial_snapshot(sim, "(day 30)")
    fig1.savefig("output_spatial.png", dpi=150)

    fig2 = plot_immune_snapshot(sim, "(day 30)")
    fig2.savefig("output_immune.png", dpi=150)

    fig3 = plot_signaling_snapshot(sim, "(day 30)")
    fig3.savefig("output_signaling.png", dpi=150)

    fig4 = plot_growth_curve(history)
    fig4.savefig("output_growth.png", dpi=150)

    fig5 = plot_immune_timeseries(history)
    fig5.savefig("output_immune_timeseries.png", dpi=150)

    fig6 = plot_signaling_timeseries(history)
    fig6.savefig("output_signaling_timeseries.png", dpi=150)

    print("\nSaved 6 output figures.")
    plt.show()