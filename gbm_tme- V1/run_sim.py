import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.engine import SimulationEngine
from visualization.plotter import plot_spatial_snapshot, plot_growth_curve
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Initializing GBM TME simulation...")
    sim = SimulationEngine()

    print("Running...")
    history = sim.run()

    print(f"Done. Final cell count: {len(sim.cells)}")

    fig1 = plot_spatial_snapshot(sim, title="(final)")
    fig1.savefig("output_spatial.png", dpi=150)

    fig2 = plot_growth_curve(history)
    fig2.savefig("output_growth.png", dpi=150)

    plt.show()
    print("Figures saved.")