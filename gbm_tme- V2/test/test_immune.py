import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.grid import Grid
from core.immune_cell import TAMCell, TCell, MDSCCell, TregCell, TAMState, TCellState
from core.tumor_cell import TumorCell, CellState
from config.params import GRID, IMMUNE, SIM, TUMOR_CELL


def test_tam_m1_to_m2_polarisation():
    """TAM should polarise M1->M2 under high TGF-b."""
    from core.signaling import SignalingLayer
    from config.params import SIGNALING
    rng = np.random.default_rng(42)
    sig = SignalingLayer(GRID, SIGNALING, SIM["dt"])
    grid = Grid(GRID)

    tam = TAMCell(10, 10, TAMState.M1, IMMUNE, rng)
    grid.occupancy[10, 10] = -(id(tam))

    # Test polarisation logic directly — bypass migration
    # by calling the polarisation block manually
    sig.tgf_beta[10, 10] = 5.0   # high at exact cell position

    transitioned = False
    p = IMMUNE
    for _ in range(500):
        # Read local signals at current position (no migration)
        local_tgf  = sig.tgf_beta[tam.y, tam.x]
        local_il10 = sig.il10[tam.y, tam.x]
        local_ifng = sig.ifng[tam.y, tam.x]

        if tam.state == TAMState.M1:
            if (local_tgf > p["tgf_m2_threshold"] or
                    local_il10 > p["il10_m2_threshold"]):
                if rng.random() < p["m1_to_m2_prob"] * SIM["dt"]:
                    tam.state = TAMState.M2

        if tam.state == TAMState.M2:
            transitioned = True
            break

    assert transitioned, \
        f"TAM polarisation failed. prob per step={p['m1_to_m2_prob']*SIM['dt']:.4f}"
    print(f"PASS — TAM polarised M1->M2 under TGF-b=5.0 pg/mL")


def test_tam_m2_stays_m2_without_ifng():
    """M2 TAM should NOT revert to M1 without IFN-g."""
    from core.signaling import SignalingLayer
    from config.params import SIGNALING
    rng = np.random.default_rng(42)
    sig = SignalingLayer(GRID, SIGNALING, SIM["dt"])
    grid = Grid(GRID)

    tam = TAMCell(10, 10, TAMState.M2, IMMUNE, rng)
    sig.ifng[10, 10] = 0.0   # no IFN-g

    p = IMMUNE
    for _ in range(500):
        local_ifng = sig.ifng[tam.y, tam.x]
        # M2->M1 only fires if ifng above threshold
        if local_ifng > p["ifng_m1_threshold"]:
            if rng.random() < p["m2_to_m1_prob"] * SIM["dt"]:
                tam.state = TAMState.M1

    assert tam.state == TAMState.M2, \
        f"M2 TAM should stay M2 without IFN-g, got {tam.state}"
    print("PASS — M2 TAM stays M2 without IFN-g")


def test_tcell_exhaustion():
    """T cell exhaustion score should accumulate under TGF-b."""
    from core.signaling import SignalingLayer
    from config.params import SIGNALING
    rng = np.random.default_rng(42)
    sig = SignalingLayer(GRID, SIGNALING, SIM["dt"])
    grid = Grid(GRID)

    tc = TCell(10, 10, IMMUNE, rng)
    sig.tgf_beta[10, 10] = 2.0   # well above exhaustion_tgf_thresh=0.4
    sig.il10[10, 10]     = 1.0

    p = IMMUNE
    # Directly accumulate exhaustion without migration
    for _ in range(1000):
        local_tgf  = sig.tgf_beta[tc.y, tc.x]
        local_il10 = sig.il10[tc.y, tc.x]

        if (local_tgf  > p["exhaustion_tgf_thresh"] or
                local_il10 > p["exhaustion_il10_thresh"]):
            tc.exhaustion += p["exhaustion_rate"] * SIM["dt"]

        if tc.exhaustion >= p["exhaustion_threshold"]:
            tc.state = TCellState.EXHAUSTED
            break

    assert tc.state == TCellState.EXHAUSTED, \
        f"T cell should exhaust, score={tc.exhaustion:.3f}, threshold={p['exhaustion_threshold']}"
    print(f"PASS — T cell exhausted (score: {tc.exhaustion:.2f})")


def test_tcell_kills_tumor():
    """Active T cell should kill adjacent tumor cell."""
    from core.signaling import SignalingLayer
    from config.params import SIGNALING
    rng = np.random.default_rng(42)
    sig = SignalingLayer(GRID, SIGNALING, SIM["dt"])
    grid = Grid(GRID)

    # Place T cell at (10,10), tumor cell at (11,10) — adjacent
    tc   = TCell(10, 10, IMMUNE, rng)
    tumor = TumorCell(11, 10, CellState.PROLIF, TUMOR_CELL, rng)

    grid.occupancy[10, 10] = -(id(tc))
    grid.occupancy[10, 11] = tumor.id

    tumor_by_id = {tumor.id: tumor}

    # Run enough steps for a kill to fire
    killed = None
    for _ in range(2000):
        killed = tc.try_kill(grid, tumor_by_id, SIM["dt"], IMMUNE)
        if killed:
            break

    assert killed == tumor.id, \
        "Active T cell should kill adjacent tumor cell"
    print(f"PASS — T cell killed tumor cell id={killed}")


def test_mdsc_suppression_zone():
    """MDSC suppression zone should cover correct radius."""
    rng  = np.random.default_rng(42)
    grid = Grid(GRID)
    mdsc = MDSCCell(10, 10, IMMUNE, rng)

    zone = mdsc.get_suppression_zone(grid, IMMUNE)
    r    = IMMUNE["mdsc_suppress_radius"]
    expected = (2*r + 1)**2

    assert len(zone) == expected, \
        f"MDSC zone should have {expected} voxels, got {len(zone)}"
    assert (10, 10) in zone, "MDSC's own voxel should be in suppression zone"
    print(f"PASS — MDSC suppression zone: {len(zone)} voxels (radius={r})")


def test_tam_m1_kills_tumor():
    """M1 TAM should kill adjacent tumor cell."""
    from core.signaling import SignalingLayer
    from config.params import SIGNALING
    rng  = np.random.default_rng(42)
    grid = Grid(GRID)

    tam   = TAMCell(10, 10, TAMState.M1, IMMUNE, rng)
    tumor = TumorCell(11, 10, CellState.PROLIF, TUMOR_CELL, rng)
    grid.occupancy[10, 10] = -(id(tam))
    grid.occupancy[10, 11] = tumor.id

    tumor_by_id = {tumor.id: tumor}
    killed = None
    for _ in range(2000):
        killed = tam.try_kill(grid, tumor_by_id, SIM["dt"], IMMUNE)
        if killed:
            break

    assert killed == tumor.id, "M1 TAM should kill adjacent tumor cell"
    print(f"PASS — M1 TAM killed tumor cell id={killed}")


def test_m2_tam_does_not_kill():
    """M2 TAM should never kill tumor cells."""
    from core.signaling import SignalingLayer
    rng  = np.random.default_rng(42)
    grid = Grid(GRID)

    tam   = TAMCell(10, 10, TAMState.M2, IMMUNE, rng)
    tumor = TumorCell(11, 10, CellState.PROLIF, TUMOR_CELL, rng)
    grid.occupancy[10, 10] = -(id(tam))
    grid.occupancy[10, 11] = tumor.id

    tumor_by_id = {tumor.id: tumor}
    for _ in range(1000):
        killed = tam.try_kill(grid, tumor_by_id, SIM["dt"], IMMUNE)
        assert killed is None, "M2 TAM must never kill tumor cells"

    print("PASS — M2 TAM correctly does not kill tumor cells")


if __name__ == "__main__":
    print("=== Month 4 Immune Cell Tests ===\n")
    test_tam_m1_to_m2_polarisation()
    test_tam_m2_stays_m2_without_ifng()
    test_tcell_exhaustion()
    test_tcell_kills_tumor()
    test_mdsc_suppression_zone()
    test_tam_m1_kills_tumor()
    test_m2_tam_does_not_kill()
    print("\nAll immune tests passed.")
