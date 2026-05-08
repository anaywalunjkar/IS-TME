import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.engine import SimulationEngine
from config.params import OXYGEN
import numpy as np

sim = SimulationEngine()

print("\nRunning 5 days...")
steps_5days = int(24 * 5 / 0.05)
for i in range(steps_5days):
    sim.step()

print("\n--- After 5 days ---")
print(f"Cell count: {len(sim.cells)}")
print(f"O2 min:     {sim.grid.oxygen.min():.4f} mmHg")
print(f"O2 mean:    {sim.grid.oxygen.mean():.4f} mmHg")
print(f"O2 at centre (100,100): {sim.grid.oxygen[100,100]:.4f} mmHg")
print(f"Hypoxia threshold:      {OXYGEN['hypoxia_thresh']} mmHg")
print(f"Necrosis threshold:     {OXYGEN['necrosis_thresh']} mmHg")
print(f"Voxels below hypoxia (< 5.0):  {(sim.grid.oxygen < 5.0).sum()}")
print(f"Voxels below necrosis (< 1.0): {(sim.grid.oxygen < 1.0).sum()}")
print(f"Voxels occupied by cells:      {(sim.grid.occupancy >= 0).sum()}")

states = {}
for c in sim.cells:
    states[c.state.name] = states.get(c.state.name, 0) + 1
print(f"Cell states: {states}")

# Check consumption map directly
print("\n--- Consumption map check ---")
consume_map = sim.o2_solver.build_consumption_map(
    sim.grid, sim.cells,
    {"GSC":       OXYGEN["consume_quiesc"],
     "PROLIF":    OXYGEN["consume_prolif"],
     "INVASIVE":  OXYGEN["consume_prolif"] * 0.7,
     "QUIESCENT": OXYGEN["consume_quiesc"],
     "NECROTIC":  0.0}
)
print(f"consume_prolif value:    {OXYGEN['consume_prolif']}")
print(f"Consumption map max:     {consume_map.max():.4f}")
print(f"Consumption map sum:     {consume_map.sum():.4f}")
print(f"Non-zero consume voxels: {(consume_map > 0).sum()}")

if consume_map.max() == 0:
    print("\nBUG FOUND: Consumption map is all zeros!")
    print("Cell state names found:")
    names = set(c.state.name for c in sim.cells)
    print(f"  {names}")
    print("Keys used in consumption dict:")
    print("  {'GSC', 'PROLIF', 'INVASIVE', 'QUIESCENT', 'NECROTIC'}")
else:
    print("\nConsumption map looks correct.")
    total_expected_drop = consume_map.sum() * 0.05
    print(f"Expected O2 drop per step: {total_expected_drop:.4f} mmHg total across grid")