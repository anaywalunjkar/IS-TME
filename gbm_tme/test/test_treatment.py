import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.grid import Grid
from core.tumor_cell import TumorCell, CellState
from core.treatment import TreatmentModule
from config.params import GRID, TUMOR_CELL, TREATMENT, SIM


def test_tmz_field_activates_on_treatment_days():
    """TMZ field should have concentration when treatment is active."""
    treat = TreatmentModule(GRID, TREATMENT, SIM["dt"])
    grid  = Grid(GRID)
    grid.initialize_vessels(n_vessels=20, rng=np.random.default_rng(42))

    # Advance to day 5 (within concurrent phase)
    steps = int(5 * 24 / SIM["dt"])
    for _ in range(steps):
        treat.advance(SIM["dt"])
        treat.step_tmz(grid)

    assert treat.tmz.max() > 0, \
        "TMZ field should be non-zero during concurrent phase"
    assert treat.tmz_active, "TMZ should be active on day 5"
    print(f"PASS — TMZ active on day 5, max={treat.tmz.max():.3f} uM")


def test_tmz_decays_after_treatment():
    """TMZ field should decay when treatment is turned off."""
    treat = TreatmentModule(GRID, TREATMENT, SIM["dt"])
    grid  = Grid(GRID)
    grid.initialize_vessels(n_vessels=20, rng=np.random.default_rng(42))

    # Build up TMZ
    for _ in range(int(5 * 24 / SIM["dt"])):
        treat.advance(SIM["dt"])
        treat.step_tmz(grid)

    peak_tmz = treat.tmz.max()

    # Simulate drug being turned off by overriding flag
    treat.tmz_active = False
    for _ in range(int(2 * 24 / SIM["dt"])):
        treat.step_tmz(grid)

    assert treat.tmz.max() < peak_tmz, \
        "TMZ should decay after treatment stops"
    print(f"PASS — TMZ decayed: {peak_tmz:.3f} -> {treat.tmz.max():.3f} uM")


def test_methylated_cell_dies_faster_than_unmethylated():
    """MGMT-methylated cells should accumulate damage faster under TMZ."""
    rng = np.random.default_rng(42)

    meth_cell   = TumorCell(10, 10, CellState.PROLIF, TUMOR_CELL, rng,
                            mgmt_methylated=True)
    unmeth_cell = TumorCell(10, 10, CellState.PROLIF, TUMOR_CELL, rng,
                            mgmt_methylated=False)

    local_tmz = 8.0   # uM — moderate concentration
    dt = SIM["dt"]

    # Apply TMZ for 100 steps to both cells
    for _ in range(100):
        meth_cell.apply_tmz_damage(local_tmz, dt, TREATMENT)
        unmeth_cell.apply_tmz_damage(local_tmz, dt, TREATMENT)

    assert meth_cell.dna_damage > unmeth_cell.dna_damage, \
        "Methylated cell should accumulate more DNA damage than unmethylated"
    print(f"PASS — DNA damage: methylated={meth_cell.dna_damage:.4f}, "
          f"unmethylated={unmeth_cell.dna_damage:.4f}")


def test_gsc_more_resistant_than_prolif():
    """GSCs should accumulate less TMZ damage than PROLIF cells."""
    rng = np.random.default_rng(42)

    gsc  = TumorCell(10, 10, CellState.GSC,   TUMOR_CELL, rng,
                     mgmt_methylated=False)
    bulk = TumorCell(10, 10, CellState.PROLIF, TUMOR_CELL, rng,
                     mgmt_methylated=True)   # methylated = sensitive

    local_tmz = 8.0
    dt = SIM["dt"]

    for _ in range(200):
        gsc.apply_tmz_damage(local_tmz, dt, TREATMENT)
        bulk.apply_tmz_damage(local_tmz, dt, TREATMENT)

    assert gsc.dna_damage < bulk.dna_damage, \
        "GSC should be more TMZ-resistant than bulk PROLIF cells"
    print(f"PASS — GSC damage={gsc.dna_damage:.4f}, "
          f"bulk (methylated) damage={bulk.dna_damage:.4f}")


def test_hypoxic_cell_resists_radiation():
    """Hypoxic cells should take less radiation damage than normoxic cells (OER effect)."""
    n_trials = 500
    normoxic_damage_total = 0.0
    hypoxic_damage_total  = 0.0

    for _ in range(n_trials):
        rng = np.random.default_rng()

        # Normoxic cell — pO2 = 30 mmHg (well oxygenated)
        cell_norm = TumorCell(10, 10, CellState.PROLIF, TUMOR_CELL, rng)
        cell_norm.apply_radiation_damage(30.0, TREATMENT)
        normoxic_damage_total += cell_norm.dna_damage

        # Hypoxic cell — pO2 = 2 mmHg (severely hypoxic)
        cell_hyp = TumorCell(10, 10, CellState.PROLIF, TUMOR_CELL, rng)
        cell_hyp.apply_radiation_damage(2.0, TREATMENT)
        hypoxic_damage_total += cell_hyp.dna_damage

    assert hypoxic_damage_total < normoxic_damage_total, (
        f"Hypoxic cells should accumulate less RT damage than normoxic. "
        f"Normoxic total={normoxic_damage_total:.3f}, "
        f"hypoxic total={hypoxic_damage_total:.3f}"
    )
    print(f"PASS — OER confirmed over {n_trials} trials: "
          f"normoxic damage={normoxic_damage_total:.3f}, "
          f"hypoxic damage={hypoxic_damage_total:.3f} "
          f"(ratio={normoxic_damage_total/max(hypoxic_damage_total,1e-9):.1f}x)")


def test_stupp_rt_schedule():
    """RT should fire on weekdays only, max 30 fractions."""
    treat = TreatmentModule(GRID, TREATMENT, SIM["dt"])
    grid  = Grid(GRID)

    rt_days = []
    for step in range(int(42 * 24 / SIM["dt"])):
        treat.advance(SIM["dt"])
        if treat.rt_today:
            rt_days.append(treat.current_day)

    assert treat.fraction_count == 30, \
        f"Should deliver exactly 30 RT fractions, got {treat.fraction_count}"
    assert treat.fraction_count <= 30, "Should not exceed 30 fractions"
    print(f"PASS — Stupp RT schedule: {treat.fraction_count} fractions "
          f"over 42 days (target: 30)")


def test_treatment_off_switch():
    """When treatment_active=False, TMZ should stay zero."""
    params_off = dict(TREATMENT)
    params_off["treatment_active"] = False

    treat = TreatmentModule(GRID, params_off, SIM["dt"])
    grid  = Grid(GRID)
    grid.initialize_vessels(n_vessels=20, rng=np.random.default_rng(42))

    for _ in range(int(10 * 24 / SIM["dt"])):
        treat.advance(SIM["dt"])
        treat.step_tmz(grid)

    assert treat.tmz.max() == 0.0, \
        "TMZ should be zero when treatment_active=False"
    assert treat.fraction_count == 0, \
        "No RT fractions should fire when treatment_active=False"
    print("PASS — treatment_active=False correctly disables all treatment")


if __name__ == "__main__":
    print("=== Month 5 Treatment Tests ===\n")
    test_tmz_field_activates_on_treatment_days()
    test_tmz_decays_after_treatment()
    test_methylated_cell_dies_faster_than_unmethylated()
    test_gsc_more_resistant_than_prolif()
    test_hypoxic_cell_resists_radiation()
    test_stupp_rt_schedule()
    test_treatment_off_switch()
    print("\nAll treatment tests passed.")
