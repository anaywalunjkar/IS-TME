import numpy as np
from core.diffusion import DiffusionSolver


class TreatmentModule:
    """
    Month 5 — Stupp Protocol Treatment Module.

    Manages:
    1. TMZ pharmacokinetics — PDE diffusion field for drug concentration
    2. Radiation therapy — linear-quadratic damage applied on scheduled days
    3. Stupp protocol scheduler — determines which treatment is active
       on each simulated day

    Stupp Protocol (concurrent phase, days 1-42):
      - RT: 2 Gy/fraction, 5 days/week (Mon-Fri), 30 fractions total
      - TMZ: 75 mg/m²/day continuously during RT
    Reference: Stupp et al. 2005 (NEJM 352:987-996)

    TMZ PK reference: Portnow et al. 2009; Ostermann et al. 2004
    Radiobiology: Hall & Giaccia 2019
    MGMT: Hegi et al. 2005
    """

    def __init__(self, grid_params, treat_params, dt):
        W = grid_params["width"]
        H = grid_params["height"]

        self.p  = treat_params
        self.dt = dt

        # ── TMZ concentration field ───────────────────────────────
        self.tmz = np.zeros((H, W))   # uM

        self.tmz_solver = DiffusionSolver(
            D  = treat_params["tmz_D"],
            dx = grid_params["voxel_size"],
            dt = dt
        )

        # ── Protocol state ────────────────────────────────────────
        self.current_day    = 0.0    # fractional day counter
        self.tmz_active     = False  # is TMZ being administered now?
        self.rt_today       = False  # is radiation given today?
        self.fraction_count = 0      # RT fractions delivered so far
        self.time_elapsed   = 0.0    # total hours elapsed

        # Track which days RT has fired (avoid double-firing)
        self._last_rt_day   = -1

        print(f"TreatmentModule initialized.")
        print(f"  Scenario: MGMT {treat_params['mgmt_scenario']}")
        print(f"  Treatment active: {treat_params['treatment_active']}")

    # ------------------------------------------------------------------
    # Protocol scheduler
    # ------------------------------------------------------------------

    def advance(self, dt):
        """
        Advance the treatment clock by dt hours.
        Updates tmz_active and rt_today flags.
        Called once per simulation step by engine.
        """
        self.time_elapsed += dt
        self.current_day   = self.time_elapsed / 24.0

        if not self.p["treatment_active"]:
            self.tmz_active = False
            self.rt_today   = False
            return

        start = self.p["concurrent_start_day"]
        end   = self.p["concurrent_end_day"]
        day   = self.current_day

        # ── TMZ: active continuously during concurrent phase ──────
        self.tmz_active = (start <= day <= end)

        # ── RT: weekdays only (5 days/week) ──────────────────────
        # Model: RT given on days 1,2,3,4,5, 8,9,10,11,12, ...
        # (skip day 6,7 = weekend, skip 13,14, etc.)
        self.rt_today = False
        if start <= day <= end:
            int_day     = int(day)
            week_number = (int_day - 1) // 7
            day_of_week = (int_day - 1) % 7   # 0=Mon ... 6=Sun
            is_weekday  = day_of_week < 5

            # Fire RT exactly once per eligible day
            if (is_weekday and
                    int_day != self._last_rt_day and
                    self.fraction_count < 30):
                self.rt_today     = True
                self._last_rt_day = int_day
                self.fraction_count += 1
            else:
                self.rt_today = False

    # ------------------------------------------------------------------
    # TMZ field update
    # ------------------------------------------------------------------

    def step_tmz(self, grid):
        """
        Advance TMZ diffusion field one timestep.

        When TMZ is active: vessels act as sources at tmz_vessel_conc.
        When TMZ is off: no source, field decays naturally.

        BBB effect is already encoded in tmz_vessel_conc (20% of plasma).
        """
        if self.tmz_active:
            source_mask = grid.vessels
            source_conc = self.p["tmz_vessel_conc"]
        else:
            # No source — drug washes out via decay
            source_mask = np.zeros(self.tmz.shape, dtype=bool)
            source_conc = 0.0

        # No cellular consumption of TMZ (it's not metabolised locally)
        empty_consume = np.zeros(self.tmz.shape)

        self.tmz_solver.step(
            field           = self.tmz,
            source_mask     = source_mask,
            source_conc     = source_conc,
            consumption_map = empty_consume,
            decay           = self.p["tmz_decay"]
        )

        np.clip(self.tmz, 0.0, self.p["tmz_max"], out=self.tmz)

    # ------------------------------------------------------------------
    # Apply damage to cells
    # ------------------------------------------------------------------

    def apply_tmz_to_cells(self, cells, dt):
        """
        Apply TMZ DNA damage to all tumor cells each timestep.
        Reads local TMZ concentration at each cell's position.
        """
        if not self.tmz_active:
            return

        for cell in cells:
            local_tmz = self.tmz[cell.y, cell.x]
            cell.apply_tmz_damage(local_tmz, dt, self.p)

    def apply_radiation_to_cells(self, cells, o2_field):
        """
        Apply radiation damage for one fraction.
        Called by engine when rt_today is True.
        Each cell's damage depends on local O2 (OER effect).
        """
        if not self.rt_today:
            return 0

        killed = 0
        for cell in cells:
            local_o2 = o2_field[cell.y, cell.x]
            was_alive = cell.alive
            cell.apply_radiation_damage(local_o2, self.p)
            if was_alive and not cell.alive:
                killed += 1

        return killed

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------

    def get_summary(self):
        return {
            "current_day":    float(self.current_day),
            "tmz_active":     self.tmz_active,
            "rt_today":       self.rt_today,
            "fraction_count": self.fraction_count,
            "tmz_mean":       float(self.tmz.mean()),
            "tmz_max":        float(self.tmz.max()),
        }

    def get_mgmt_counts(self, cells):
        """Return count of methylated vs unmethylated cells."""
        methylated   = sum(1 for c in cells if c.mgmt_methylated)
        unmethylated = sum(1 for c in cells if not c.mgmt_methylated)
        gsc_count    = sum(1 for c in cells if c.state.name == "GSC")
        return {
            "methylated":   methylated,
            "unmethylated": unmethylated,
            "gsc":          gsc_count,
        }
