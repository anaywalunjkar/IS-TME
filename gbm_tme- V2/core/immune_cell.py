import numpy as np
from enum import Enum, auto
from core.cell import BaseCell


# ── Immune cell state enums ───────────────────────────────────────────────────

class TAMState(Enum):
    M1 = auto()   # pro-inflammatory, anti-tumor — kills tumor cells, secretes IFN-g
    M2 = auto()   # anti-inflammatory, pro-tumor  — secretes TGF-b, IL-10

class TCellState(Enum):
    ACTIVE    = auto()   # killing, secreting IFN-g
    EXHAUSTED = auto()   # functionally impaired by TGF-b and IL-10

class MDSCState(Enum):
    ACTIVE = auto()   # suppressing T cells in local neighbourhood

class TregState(Enum):
    ACTIVE = auto()   # suppressing T cells, secreting IL-10 and TGF-b


# ── TAM ──────────────────────────────────────────────────────────────────────

class TAMCell(BaseCell):
    """
    Tumor-Associated Macrophage.

    Enters through vessels, migrates toward tumor.
    Polarisation state (M1/M2) driven by local cytokine balance:
      - High TGF-b + IL-10  →  M1 polarises to M2 (immunosuppressive)
      - High IFN-g           →  M2 reverts to M1 (rare in GBM)

    In GBM the microenvironment is strongly M2-polarising:
    by day 30, >70% of TAMs should be M2.
    Reference: Hambardzumyan et al. 2016 (Nature Neuroscience)
    """

    def __init__(self, x, y, state=TAMState.M1, params=None, rng=None):
        super().__init__(x, y, cell_type="TAM")
        self.state    = state
        self.params   = params or {}
        self.rng      = rng or np.random.default_rng()
        self.age      = 0.0
        self.alive    = True

    def update(self, grid, signaling, dt, p):
        """
        Update TAM state each timestep.
        1. Age — die at lifespan
        2. Polarisation — read TGF-b, IL-10, IFN-g fields
        3. Migration — random walk toward tumor (follow TGF-b gradient)
        4. Killing — M1 only, adjacent tumor cells
        """
        self.age += dt

        if self.age > p["tam_lifespan"]:
            self.alive = False
            return

        local_tgf  = signaling.tgf_beta[self.y, self.x]
        local_il10 = signaling.il10[self.y, self.x]
        local_ifng = signaling.ifng[self.y, self.x]

        # ── Polarisation ──────────────────────────────────────────
        if self.state == TAMState.M1:
            # M1 → M2 if TGF-b OR IL-10 above threshold
            if (local_tgf  > p["tgf_m2_threshold"] or
                    local_il10 > p["il10_m2_threshold"]):
                if self.rng.random() < p["m1_to_m2_prob"] * dt:
                    self.state = TAMState.M2

        elif self.state == TAMState.M2:
            # M2 → M1 only if IFN-g high enough (rare in GBM)
            if local_ifng > p["ifng_m1_threshold"]:
                if self.rng.random() < p["m2_to_m1_prob"] * dt:
                    self.state = TAMState.M1

        # ── Migration — follow TGF-b gradient (toward tumor) ─────
        self._migrate(grid, signaling.tgf_beta, dt, p)

    def try_kill(self, grid, tumor_cells_by_pos, dt, p):
        """
        M1 TAMs attempt to kill adjacent tumor cells.
        M2 TAMs do not kill.
        Returns cell_id of killed cell, or None.
        """
        if self.state != TAMState.M1:
            return None

        neighbours = grid.get_neighbors(self.x, self.y)
        for (nx, ny) in neighbours:
            cell_id = grid.occupancy[ny, nx]
            if cell_id in tumor_cells_by_pos:
                if self.rng.random() < p["m1_kill_prob"] * dt:
                    return cell_id
        return None

    def get_secretion(self, p):
        """Return (tgf_sec, il10_sec, ifng_sec) based on current polarisation."""
        if self.state == TAMState.M1:
            return (p["m1_tgf_secretion"],
                    p["m1_il10_secretion"],
                    p["m1_ifng_secretion"])
        else:  # M2
            return (p["m2_tgf_secretion"],
                    p["m2_il10_secretion"],
                    p["m2_ifng_secretion"])

    def _migrate(self, grid, gradient_field, dt, p):
        """Biased random walk up the gradient field (toward tumor)."""
        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return
        # Move toward highest gradient value with prob 0.4
        vals = [gradient_field[ny, nx] for nx, ny in free]
        best = free[int(np.argmax(vals))]
        if self.rng.random() < 0.4:
            nx, ny = best
        else:
            nx, ny = free[self.rng.integers(len(free))]
        grid.occupancy[self.y, self.x] = -1
        self.x, self.y = nx, ny
        grid.occupancy[ny, nx] = -(id(self))   # negative = immune cell

    def __repr__(self):
        return f"TAMCell(id={id(self)}, state={self.state.name}, pos=({self.x},{self.y}))"


# ── CD8+ T Cell ───────────────────────────────────────────────────────────────

class TCell(BaseCell):
    """
    CD8+ Cytotoxic T Lymphocyte.

    Enters through vessels, migrates toward tumor cells.
    Kills adjacent PROLIF and INVASIVE tumor cells.
    Becomes EXHAUSTED under sustained TGF-b and IL-10 exposure —
    the primary mechanism by which GBM evades anti-tumor immunity.

    Exhaustion is modelled as a cumulative internal variable.
    Reference: Woroniecka et al. 2018 (Clinical Cancer Research)
    """

    def __init__(self, x, y, params=None, rng=None):
        super().__init__(x, y, cell_type="TCELL")
        self.state        = TCellState.ACTIVE
        self.params       = params or {}
        self.rng          = rng or np.random.default_rng()
        self.age          = 0.0
        self.alive        = True
        self.exhaustion   = 0.0   # cumulative exhaustion score [0, 1+]

    def update(self, grid, signaling, dt, p):
        """
        1. Age — die at lifespan
        2. Exhaustion — accumulate under TGF-b and IL-10
        3. Migration — chemotax toward tumor (up TGF-b gradient as proxy)
        """
        self.age += dt

        if self.age > p["tcell_lifespan"]:
            self.alive = False
            return

        local_tgf  = signaling.tgf_beta[self.y, self.x]
        local_il10 = signaling.il10[self.y, self.x]

        # ── Exhaustion accumulation ───────────────────────────────
        if (local_tgf  > p["exhaustion_tgf_thresh"] or
                local_il10 > p["exhaustion_il10_thresh"]):
            self.exhaustion += p["exhaustion_rate"] * dt

        # State transition
        if self.exhaustion >= p["exhaustion_threshold"]:
            self.state = TCellState.EXHAUSTED

        # ── Migration — toward tumor (up TGF-b gradient) ─────────
        self._migrate(grid, signaling.tgf_beta, dt, p)

    def try_kill(self, grid, tumor_cells_by_pos, dt, p,
                 local_suppression=1.0):
        """
        Attempt to kill adjacent tumor cell.
        local_suppression: float [0,1] — reduced by MDSCs and Tregs nearby.
        Returns cell_id of killed tumor cell, or None.
        """
        if not self.alive:
            return None

        # Killing rate scales with state and local suppression
        if self.state == TCellState.EXHAUSTED:
            kill_rate = p["tcell_kill_prob"] * p["exhaustion_kill_factor"]
        else:
            kill_rate = p["tcell_kill_prob"]

        # Apply suppression from MDSCs and Tregs
        kill_rate *= local_suppression

        neighbours = grid.get_neighbors(self.x, self.y)
        for (nx, ny) in neighbours:
            cell_id = grid.occupancy[ny, nx]
            if cell_id in tumor_cells_by_pos:
                tc = tumor_cells_by_pos[cell_id]
                # Only kill live tumor cells (not already necrotic)
                if tc.state.name not in ("NECROTIC",) and tc.alive:
                    if self.rng.random() < kill_rate * dt:
                        return cell_id
        return None

    def get_ifng_secretion(self, p):
        """Active T cells secrete IFN-g; exhausted cells secrete very little."""
        if self.state == TCellState.ACTIVE:
            return p["tcell_ifng_secretion"]
        return p["tcell_ifng_secretion"] * 0.05   # exhausted: near-zero

    def _migrate(self, grid, gradient_field, dt, p):
        """Faster migration than TAMs; stronger chemotaxis bias."""
        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return
        vals = [gradient_field[ny, nx] for nx, ny in free]
        best = free[int(np.argmax(vals))]
        if self.rng.random() < 0.6:   # stronger chemotaxis than TAMs
            nx, ny = best
        else:
            nx, ny = free[self.rng.integers(len(free))]
        grid.occupancy[self.y, self.x] = -1
        self.x, self.y = nx, ny
        grid.occupancy[ny, nx] = -(id(self))

    def __repr__(self):
        return (f"TCell(id={id(self)}, state={self.state.name}, "
                f"exhaustion={self.exhaustion:.2f}, pos=({self.x},{self.y}))")


# ── MDSC ─────────────────────────────────────────────────────────────────────

class MDSCCell(BaseCell):
    """
    Myeloid-Derived Suppressor Cell.

    Does not kill tumor cells directly.
    Suppresses T cell killing rate within a radius.
    Recruited by VEGF and tumor-secreted factors.
    Reference: Gielen et al. 2015 (Cancer Research)
    """

    def __init__(self, x, y, params=None, rng=None):
        super().__init__(x, y, cell_type="MDSC")
        self.state  = MDSCState.ACTIVE
        self.params = params or {}
        self.rng    = rng or np.random.default_rng()
        self.age    = 0.0
        self.alive  = True

    def update(self, grid, signaling, dt, p):
        self.age += dt
        if self.age > p["mdsc_lifespan"]:
            self.alive = False
            return
        # Slow random migration
        self._migrate(grid, dt)

    def get_suppression_zone(self, grid, p):
        """
        Returns list of (x,y) voxels within suppression radius.
        T cells in these voxels have their killing rate multiplied
        by mdsc_suppress_factor.
        """
        r = p["mdsc_suppress_radius"]
        zone = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = self.x + dx, self.y + dy
                if grid.in_bounds(nx, ny):
                    zone.append((nx, ny))
        return zone

    def _migrate(self, grid, dt):
        """Pure random walk — MDSCs are not strongly directed."""
        if self.rng.random() > 0.3 * dt:
            return
        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return
        nx, ny = free[self.rng.integers(len(free))]
        grid.occupancy[self.y, self.x] = -1
        self.x, self.y = nx, ny
        grid.occupancy[ny, nx] = -(id(self))

    def __repr__(self):
        return f"MDSCCell(id={id(self)}, pos=({self.x},{self.y}))"


# ── Treg ─────────────────────────────────────────────────────────────────────

class TregCell(BaseCell):
    """
    Regulatory T Cell.

    Suppresses CD8+ T cell activity in local neighbourhood.
    Also secretes IL-10 and TGF-b, reinforcing immunosuppression.
    Smaller suppression radius than MDSC but stronger per-cell effect.
    """

    def __init__(self, x, y, params=None, rng=None):
        super().__init__(x, y, cell_type="TREG")
        self.state  = TregState.ACTIVE
        self.params = params or {}
        self.rng    = rng or np.random.default_rng()
        self.age    = 0.0
        self.alive  = True

    def update(self, grid, signaling, dt, p):
        self.age += dt
        if self.age > p["treg_lifespan"]:
            self.alive = False
            return
        self._migrate(grid, signaling.tgf_beta, dt, p)

    def get_suppression_zone(self, grid, p):
        """Voxels within Treg suppression radius."""
        r = p["treg_suppress_radius"]
        zone = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = self.x + dx, self.y + dy
                if grid.in_bounds(nx, ny):
                    zone.append((nx, ny))
        return zone

    def get_secretion(self, p):
        """Returns (tgf_sec, il10_sec)."""
        return (p["treg_tgf_secretion"], p["treg_il10_secretion"])

    def _migrate(self, grid, gradient_field, dt, p):
        """Follow TGF-b gradient like TAMs."""
        free = grid.free_neighbors(self.x, self.y)
        if not free:
            return
        vals = [gradient_field[ny, nx] for nx, ny in free]
        best = free[int(np.argmax(vals))]
        if self.rng.random() < 0.4:
            nx, ny = best
        else:
            nx, ny = free[self.rng.integers(len(free))]
        grid.occupancy[self.y, self.x] = -1
        self.x, self.y = nx, ny
        grid.occupancy[ny, nx] = -(id(self))

    def __repr__(self):
        return f"TregCell(id={id(self)}, pos=({self.x},{self.y}))"
