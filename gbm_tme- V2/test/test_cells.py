# tests/test_cells.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid import Grid
from core.tumor_cell import TumorCell, CellState
from config.params import GRID, TUMOR_CELL, SIM

def test_grid_init():
    grid = Grid(GRID)
    rng = np.random.default_rng(42)
    grid.initialize_vessels(n_vessels=20, rng=rng)
    assert grid.vessels.sum() == 20, "Wrong vessel count"
    assert grid.occupancy.max() == -1, "Occupancy should start empty"
    print(f"PASS — grid {grid.W}x{grid.H}, 20 vessels placed")

def test_cell_state_transition():
    """Cell should go INVASIVE under hypoxia and NECROTIC under anoxia."""
    rng = np.random.default_rng(42)
    cell = TumorCell(10, 10, CellState.PROLIF, TUMOR_CELL, rng)

    # With prob=0.3 and dt=0.05, expected steps to transition = 1/(0.3*0.05) = ~67
    # Run 300 steps to be statistically certain (~99.9% chance of transition)
    for _ in range(300):
        if cell.state == CellState.INVASIVE:
            break
        cell.update_state(
            local_o2=2.0,   # below hypoxia_thresh (5.0)
            local_glc=5.0,
            dt=SIM["dt"],
            p=TUMOR_CELL
        )

    assert cell.state == CellState.INVASIVE, \
        f"Expected INVASIVE after 300 hypoxic steps, got {cell.state}. " \
        f"Check hypoxia_to_invasive_prob in params.py"
    print(f"PASS — cell transitioned to INVASIVE under hypoxia")

    # Simulate anoxia — should be immediate (single step)
    cell.update_state(local_o2=0.5, local_glc=5.0,
                      dt=SIM["dt"], p=TUMOR_CELL)
    assert cell.state == CellState.NECROTIC, \
        f"Expected NECROTIC under anoxia, got {cell.state}"
    print(f"PASS — cell transitioned to NECROTIC under anoxia")

def test_cell_division():
    """Cell should produce a daughter when cycle clock has elapsed."""
    rng = np.random.default_rng(42)
    grid = Grid(GRID)

    cell = TumorCell(100, 100, CellState.PROLIF, TUMOR_CELL, rng)
    grid.occupancy[100, 100] = cell.id

    # Force clock to just before division threshold
    cell.cycle_clock = TUMOR_CELL["prolif_cycle_time"] - SIM["dt"]

    daughter = cell.try_divide(grid, SIM["dt"], TUMOR_CELL)
    assert daughter is not None, \
        "Division should have occurred — check try_divide logic"
    assert daughter.state == CellState.PROLIF
    print(f"PASS — division produced daughter cell id={daughter.id} "
          f"at ({daughter.x}, {daughter.y})")

if __name__ == "__main__":
    print("=== Cell & grid tests ===")
    test_grid_init()
    test_cell_state_transition()
    test_cell_division()
    print("\nAll cell tests passed.")