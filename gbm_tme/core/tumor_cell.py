import numpy as np
from enum import Enum, auto


class CellState(Enum):
    GSC        = auto()
    PROLIF     = auto()
    INVASIVE   = auto()
    QUIESCENT  = auto()
    NECROTIC   = auto()


class TumorCell:
    _id_counter = 0

    def __init__(self, x, y, state=CellState.PROLIF,
                 params=None, rng=None, mgmt_methylated=None):
        TumorCell._id_counter += 1
        self.id    = TumorCell._id_counter
        self.x     = x
        self.y     = y
        self.state = state
        self.params = params
        self.rng   = rng or np.random.default_rng()

        # Internal clocks
        self.age            = 0.0
        self.cycle_clock    = 0.0
        self.hypoxia_time   = 0.0
        self.starve_time    = 0.0
        self.necrotic_timer = 0.0

        self.alive = True

        # ── Month 5: MGMT methylation status ─────────────────────
        # GSCs are always unmethylated (treatment resistant)
        # For other cells: set by engine based on scenario
        if self.state == CellState.GSC:
            self.mgmt_methylated = False   # GSCs always resistant
        elif mgmt_methylated is not None:
            self.mgmt_methylated = mgmt_methylated
        else:
            self.mgmt_methylated = False   # default — set by engine at seeding

        # ── Month 5: DNA damage accumulation ─────────────────────
        # Accumulates from TMZ and radiation each timestep
        # When it exceeds 1.0 → cell dies (treatment-induced necrosis)
        self.dna_damage = 0.0

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def update_state(self, local_o2, local_glc, dt, p):
        if self.state == CellState.NECROTIC:
            self.necrotic_timer += dt
            if self.necrotic_timer > 48.0:
                self.alive = False
            return

        hypoxic = local_o2  < p["hypoxia_thresh"]
        anoxic  = local_o2  < p["necrosis_thresh"]
        starved = local_glc < p["starvation_thresh"]

        if hypoxic:
            self.hypoxia_time += dt
        else:
            self.hypoxia_time = max(0.0, self.hypoxia_time - dt)

        if starved:
            self.starve_time += dt
        else:
            self.starve_time = 0.0

        # Necrosis from microenvironment
        # GSCs are stress-tolerant — survive hypoxia via HIF pathways
        # Reference: Lathia et al. 2015 (Genes & Development)
        if self.state == CellState.GSC:
            # GSCs only die if BOTH fully anoxic AND starved for 2x longer
            if anoxic and self.starve_time > p["starvation_to_necrosis_time"] * 2:
                self._become_necrotic()
                return
        else:
            if anoxic or self.starve_time > p["starvation_to_necrosis_time"]:
                self._become_necrotic()
                return

        # Hypoxia → invasive
        if hypoxic and self.state == CellState.PROLIF:
            if self.rng.random() < p["hypoxia_to_invasive_prob"] * dt:
                self.state = CellState.INVASIVE
                return

        # Recovery
        if not hypoxic and self.state == CellState.INVASIVE:
            if self.rng.random() < p.get("invasive_revert_prob", 0.001) * dt:
                self.state = CellState.PROLIF
                return

        self.age += dt

    # ------------------------------------------------------------------
    # Month 5: Treatment damage methods
    # ------------------------------------------------------------------

    def apply_tmz_damage(self, local_tmz, dt, treat_p):
        """
        Apply TMZ-induced DNA damage each timestep.

        Damage rate = tmz_kill_base × local_tmz concentration
        MGMT-methylated cells: full damage (cannot repair)
        MGMT-unmethylated cells: damage reduced by mgmt_repair_factor
        GSCs: additional resistance via gsc_tmz_resistance

        If cumulative dna_damage >= 1.0 → cell becomes necrotic.

        Reference: Portnow et al. 2009 (Clinical Cancer Research)
        """
        if self.state == CellState.NECROTIC or not self.alive:
            return
        if local_tmz <= 0:
            return

        # Base damage rate
        damage_rate = treat_p["tmz_kill_base"] * local_tmz * dt

        # GSC resistance (always unmethylated + activated DNA checkpoints)
        if self.state == CellState.GSC:
            damage_rate *= treat_p["gsc_tmz_resistance"]
        elif not self.mgmt_methylated:
            # Unmethylated: MGMT repairs most damage
            damage_rate *= treat_p["mgmt_repair_factor"]
        # Methylated: full damage (no reduction)

        self.dna_damage += damage_rate

        if self.dna_damage >= 1.0:
            self._become_necrotic()

    def apply_radiation_damage(self, local_o2, treat_p):
        """
        Apply radiation damage for one fraction (2 Gy).

        Uses linear-quadratic model:
            SF = exp(-alpha*D - beta*D²)
        where SF = survival fraction for that fraction.

        Oxygen Enhancement Ratio (OER) reduces effective dose in hypoxia:
            OER = oer_max * K / (pO2 + K)
            effective_dose = dose / OER

        GSCs have additional radioresistance (Bao et al. 2006).

        Called by engine on each radiation day — not every timestep.
        """
        if self.state == CellState.NECROTIC or not self.alive:
            return

        dose = treat_p["rt_dose_per_fraction"]

        # Oxygen Enhancement Ratio — hypoxic cells more resistant
        oer_max = treat_p["rt_oer_max"]
        k       = treat_p["rt_oer_k"]
        oer     = oer_max * k / (local_o2 + k)
        oer     = max(1.0, oer)   # OER >= 1 always

        # Effective dose corrected for hypoxia
        effective_dose = dose / oer

        # Linear-quadratic survival fraction
        alpha = treat_p["rt_alpha"]
        beta  = treat_p["rt_beta"]
        D     = effective_dose
        sf    = np.exp(-alpha * D - beta * D * D)

        # GSC additional radioresistance
        if self.state == CellState.GSC:
            sf = sf ** treat_p["rt_gsc_resistance"]

        # Probabilistic death — cell dies with prob (1 - sf)
        if self.rng.random() > sf:
            self.dna_damage += 0.5   # partial damage — may accumulate
            if self.dna_damage >= 1.0:
                self._become_necrotic()

    def _become_necrotic(self):
        """Transition to necrotic state — shared by all death pathways."""
        self.state          = CellState.NECROTIC
        self.necrotic_timer = 0.0
        # alive stays True — 48hr persistence before clearing

    # ------------------------------------------------------------------
    # Division
    # ------------------------------------------------------------------

    def try_divide(self, grid, dt, p):
        if self.state not in (CellState.PROLIF, CellState.GSC):
            return None
        if self.state == CellState.NECROTIC:
            return None

        cycle_time = (p["gsc_cycle_time"] if self.state == CellState.GSC
                      else p["prolif_cycle_time"])
        self.cycle_clock += dt

        if self.cycle_clock < cycle_time:
            return None

        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return None

        self.cycle_clock = 0.0
        idx = self.rng.integers(len(free))
        dx, dy = free[idx]

        if self.state == CellState.GSC:
            sym_prob = p.get("symmetric_prob", p.get("gsc_symmetric_prob", 0.1))
            daughter_state = (CellState.GSC
                              if self.rng.random() < sym_prob
                              else CellState.PROLIF)
        else:
            daughter_state = CellState.PROLIF

        # Daughter inherits MGMT status from parent
        daughter = TumorCell(dx, dy, daughter_state, self.params,
                             self.rng, mgmt_methylated=self.mgmt_methylated)
        grid.occupancy[dy, dx] = daughter.id
        return daughter

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def try_migrate(self, grid, o2_field, dt, p):
        if self.state == CellState.NECROTIC:
            return

        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return

        if self.state == CellState.INVASIVE:
            o2_vals = [o2_field[ny, nx] for nx, ny in free]
            best_idx = int(np.argmax(o2_vals))
            if self.rng.random() < p["chemotaxis_strength"]:
                nx, ny = free[best_idx]
            else:
                nx, ny = free[self.rng.integers(len(free))]
        else:
            if self.rng.random() > 0.02 * dt:
                return
            nx, ny = free[self.rng.integers(len(free))]

        grid.occupancy[self.y, self.x] = -1
        self.x, self.y = nx, ny
        grid.occupancy[ny, nx] = self.id

    def __repr__(self):
        mgmt = "M+" if self.mgmt_methylated else "M-"
        return (f"TumorCell(id={self.id}, state={self.state.name}, "
                f"MGMT={mgmt}, dmg={self.dna_damage:.2f}, "
                f"pos=({self.x},{self.y}))")