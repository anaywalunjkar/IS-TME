# tests/test_diffusion.py
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.diffusion import DiffusionSolver
from config.params import OXYGEN, GRID, SIM

def test_stability_check():
    """Should raise if dt is too large."""
    try:
        bad_solver = DiffusionSolver(D=1800.0, dx=20.0, dt=10.0)
        print("FAIL — should have raised ValueError for unstable dt")
    except ValueError as e:
        print(f"PASS — stability check works: {e}")

def test_oxygen_decays_from_vessel():
    """O2 should spread from vessel source and decay away from it."""
    solver = DiffusionSolver(
        D=OXYGEN["D"],
        dx=GRID["voxel_size"],
        dt=SIM["dt"]
    )

    # Small 20x20 grid, single vessel at centre
    field = np.zeros((20, 20))
    source_mask = np.zeros((20, 20), dtype=bool)
    source_mask[10, 10] = True

    # Run 100 steps
    for _ in range(100):
        solver.step(
            field=field,
            source_mask=source_mask,
            source_conc=OXYGEN["vessel_conc"],
            consumption_map=np.zeros((20, 20))
        )

    # Centre should be at source concentration
    assert abs(field[10, 10] - OXYGEN["vessel_conc"]) < 0.1, \
        f"Vessel voxel wrong: {field[10,10]}"

    # Corners should be lower than centre
    assert field[0, 0] < field[10, 10], \
        "O2 should decrease away from vessel"

    # No negative values
    assert field.min() >= 0, f"Negative O2: {field.min()}"

    print(f"PASS — O2 at vessel: {field[10,10]:.1f} mmHg, "
          f"at corner: {field[0,0]:.2f} mmHg")

def test_consumption_reduces_o2():
    """Cells consuming O2 should reduce concentration vs no consumption."""
    solver = DiffusionSolver(D=OXYGEN["D"], dx=GRID["voxel_size"], dt=SIM["dt"])

    field_no_consume  = np.zeros((20, 20))
    field_with_consume = np.zeros((20, 20))
    source_mask = np.zeros((20, 20), dtype=bool)
    source_mask[10, 10] = True

    consume_map = np.zeros((20, 20))
    consume_map[8:12, 8:12] = OXYGEN["consume_prolif"]  # cells near vessel

    for _ in range(200):
        solver.step(field_no_consume,  source_mask, OXYGEN["vessel_conc"], np.zeros((20,20)))
        solver.step(field_with_consume, source_mask, OXYGEN["vessel_conc"], consume_map)

    mean_no_c = field_no_consume.mean()
    mean_with_c = field_with_consume.mean()
    assert mean_with_c < mean_no_c, "Consumption should reduce O2"
    print(f"PASS — mean O2 without consumption: {mean_no_c:.2f}, "
          f"with consumption: {mean_with_c:.2f}")

if __name__ == "__main__":
    print("=== Diffusion solver tests ===")
    test_stability_check()
    test_oxygen_decays_from_vessel()
    test_consumption_reduces_o2()
    print("\nAll diffusion tests passed.")