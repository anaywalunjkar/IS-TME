import numpy as np
from enum import Enum, auto


class CellState(Enum):
    GSC        = auto()   # glioblastoma stem cell — slow cycling, treatment resistant
    PROLIF     = auto()   # actively proliferating bulk tumor cell
    INVASIVE   = auto()   # migratory phenotype triggered by hypoxia
    QUIESCENT  = auto()   # cell cycle arrested — low O2 or glucose
    NECROTIC   = auto()   # dead — persists on grid for 48 hrs before clearing


class TumorCell:
    _id_counter = 0

    def __init__(self, x, y, state=CellState.PROLIF, params=None, rng=None):
        TumorCell._id_counter += 1
        self.id    = TumorCell._id_counter
        self.x     = x
        self.y     = y
        self.state = state
        self.params = params
        self.rng   = rng or np.random.default_rng()

        # Internal clocks
        self.age            = 0.0   # hr since birth
        self.cycle_clock    = 0.0   # hr toward next division
        self.hypoxia_time   = 0.0   # hr spent in hypoxia
        self.starve_time    = 0.0   # hr without sufficient glucose
        self.necrotic_timer = 0.0   # hr since becoming necrotic

        self.alive = True

    # ------------------------------------------------------------------
    # State transitions — called every timestep
    # ------------------------------------------------------------------

    def update_state(self, local_o2, local_glc, dt, p):
        """
        Evaluate local microenvironment and update cell phenotype.

        Args:
            local_o2:  O2 concentration at cell voxel (mmHg)
            local_glc: glucose concentration at cell voxel (mM)
            dt:        timestep (hr)
            p:         TUMOR_CELL params dict
        """

        # --- Necrotic cells persist on grid for 48 hrs then clear ---
        if self.state == CellState.NECROTIC:
            self.necrotic_timer += dt
            if self.necrotic_timer > 48.0:
                self.alive = False
            return

        # --- Evaluate microenvironment conditions ---
        hypoxic = local_o2  < p["hypoxia_thresh"]
        anoxic  = local_o2  < p["necrosis_thresh"]
        starved = local_glc < p["starvation_thresh"]

        # Accumulate stress timers
        if hypoxic:
            self.hypoxia_time += dt
        else:
            self.hypoxia_time = max(0.0, self.hypoxia_time - dt)

        if starved:
            self.starve_time += dt
        else:
            self.starve_time = 0.0

        # --- Necrosis: anoxia OR prolonged starvation ---
        if anoxic or self.starve_time > p["starvation_to_necrosis_time"]:
            self.state          = CellState.NECROTIC
            self.necrotic_timer = 0.0
            # Note: alive stays True — cell persists 48 hrs (handled above)
            return

        # --- Hypoxia → invasive phenotype ---
        if hypoxic and self.state == CellState.PROLIF:
            if self.rng.random() < p["hypoxia_to_invasive_prob"] * dt:
                self.state = CellState.INVASIVE
                return

        # --- Recovery: well-oxygenated invasive cell reverts slowly ---
        if not hypoxic and self.state == CellState.INVASIVE:
            revert_prob = p.get("invasive_revert_prob", 0.005)
            if self.rng.random() < revert_prob * dt:
                self.state = CellState.PROLIF
                return

        # --- GSC: always stays GSC unless environmental kill ---
        # (GSC state managed at division, not here)

        # --- Increment age ---
        self.age += dt

    # ------------------------------------------------------------------
    # Division
    # ------------------------------------------------------------------

    def try_divide(self, grid, dt, p):
        """
        Attempt cell division if cycle clock has elapsed and
        a free neighbour voxel is available (contact inhibition).

        Returns:
            New TumorCell (daughter) if division occurred, else None.
        """
        if self.state not in (CellState.PROLIF, CellState.GSC):
            return None

        if self.state == CellState.NECROTIC:
            return None

        cycle_time = (p["gsc_cycle_time"] if self.state == CellState.GSC
                      else p["prolif_cycle_time"])

        self.cycle_clock += dt

        if self.cycle_clock < cycle_time:
            return None

        # Find a free neighbour
        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return None   # contact inhibition — no space to divide

        self.cycle_clock = 0.0

        # Pick random free neighbour for daughter placement
        idx = self.rng.integers(len(free))
        dx, dy = free[idx]

        # GSC division: symmetric (GSC+GSC) vs asymmetric (GSC+PROLIF)
        if self.state == CellState.GSC:
            sym_prob = p.get("symmetric_prob", p.get("gsc_symmetric_prob", 0.1))
            daughter_state = (CellState.GSC
                              if self.rng.random() < sym_prob
                              else CellState.PROLIF)
        else:
            daughter_state = CellState.PROLIF

        daughter = TumorCell(dx, dy, daughter_state, self.params, self.rng)
        grid.occupancy[dy, dx] = daughter.id
        return daughter

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def try_migrate(self, grid, o2_field, dt, p):
        """
        Move cell to a neighbouring voxel.
        - INVASIVE cells: biased toward highest O2 (chemotaxis away from hypoxia)
        - PROLIF/GSC cells: rare random motility
        - NECROTIC cells: immobile
        """
        if self.state == CellState.NECROTIC:
            return

        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return

        if self.state == CellState.INVASIVE:
            # Chemotaxis: move toward highest O2 neighbour
            o2_vals = [o2_field[ny, nx] for nx, ny in free]
            best_idx = int(np.argmax(o2_vals))

            if self.rng.random() < p["chemotaxis_strength"]:
                nx, ny = free[best_idx]
            else:
                nx, ny = free[self.rng.integers(len(free))]

        else:
            # Low-probability random walk for non-invasive cells
            if self.rng.random() > 0.02 * dt:
                return
            nx, ny = free[self.rng.integers(len(free))]

        # Execute move
        grid.occupancy[self.y, self.x] = -1
        self.x, self.y = nx, ny
        grid.occupancy[ny, nx] = self.id

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"TumorCell(id={self.id}, state={self.state.name}, "
                f"pos=({self.x},{self.y}), age={self.age:.1f}hr)")