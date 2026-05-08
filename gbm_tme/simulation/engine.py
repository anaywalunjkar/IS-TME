import numpy as np
from tqdm import tqdm
from core.grid import Grid
from core.tumor_cell import TumorCell, CellState
from core.diffusion import DiffusionSolver
from core.signaling import SignalingLayer
from core.immune_cell import TAMCell, TCell, MDSCCell, TregCell, TAMState
from config.params import GRID, OXYGEN, GLUCOSE, TUMOR_CELL, SIGNALING, IMMUNE, SIM


class SimulationEngine:

    def __init__(self):
        self.rng = np.random.default_rng(SIM["seed"])
        self.grid = Grid(GRID)
        self.grid.initialize_vessels(n_vessels=40, rng=self.rng)
        self.grid.initialize_substrates(OXYGEN, GLUCOSE)

        self.o2_solver  = DiffusionSolver(OXYGEN["D"],  GRID["voxel_size"], SIM["dt"])
        self.glc_solver = DiffusionSolver(GLUCOSE["D"], GRID["voxel_size"], SIM["dt"])
        self.signaling  = SignalingLayer(GRID, SIGNALING, SIM["dt"])

        # Tumor cells
        self.cells = []

        # Month 4 — immune cell lists
        self.tams  = []   # TAMCell list
        self.tcells= []   # TCell list
        self.mdscs = []   # MDSCCell list
        self.tregs = []   # TregCell list

        self.history = []

        # Pre-equilibrate substrates
        print("Pre-equilibrating substrate fields...")
        empty_consume = np.zeros((GRID["height"], GRID["width"]))
        for _ in range(2000):
            self.o2_solver.step(self.grid.oxygen, self.grid.vessels,
                                OXYGEN["vessel_conc"], empty_consume)
            self.glc_solver.step(self.grid.glucose, self.grid.vessels,
                                 GLUCOSE["vessel_conc"], empty_consume)
        print(f"O2 after equilibration: "
              f"{self.grid.oxygen.min():.1f} - {self.grid.oxygen.max():.1f} mmHg")

        self._seed_tumor()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _seed_tumor(self):
        vy, vx  = np.where(self.grid.vessels)
        cx, cy  = GRID["width"] // 2, GRID["height"] // 2
        dists   = ((vx - cx)**2 + (vy - cy)**2)
        nearest = np.argmin(dists)
        sx, sy  = int(vx[nearest]), int(vy[nearest])

        placed, attempts = 0, 0
        while placed < SIM["n_initial_cells"] and attempts < 1000:
            x = sx + self.rng.integers(-8, 9)
            y = sy + self.rng.integers(-8, 9)
            if (self.grid.in_bounds(x, y) and
                    self.grid.occupancy[y, x] == -1 and
                    self.grid.oxygen[y, x] > OXYGEN["hypoxia_thresh"]):
                cell = TumorCell(x, y, CellState.PROLIF, TUMOR_CELL, self.rng)
                self.cells.append(cell)
                self.grid.occupancy[y, x] = cell.id
                placed += 1
            attempts += 1
        print(f"Seeded {placed} tumor cells near vessel at ({sx}, {sy})")

    def _recruit_immune_cells(self):
        """
        Each timestep, each vessel has a small probability of releasing
        each immune cell type into an adjacent free voxel.
        This models transendothelial migration (extravasation).

        TAMs dominate (5:1 ratio vs T cells) — matches GBM clinical data.
        """
        vy, vx = np.where(self.grid.vessels)
        p = IMMUNE
        dt = SIM["dt"]

        for (vxi, vyi) in zip(vx.tolist(), vy.tolist()):
            free = self.grid.free_neighbors(vxi, vyi)
            if not free:
                continue

            # TAM recruitment
            if self.rng.random() < p["tam_recruit_rate"] * dt:
                nx, ny = free[self.rng.integers(len(free))]
                tam = TAMCell(nx, ny, TAMState.M1, p, self.rng)
                self.tams.append(tam)
                self.grid.occupancy[ny, nx] = -(id(tam))

            # T cell recruitment
            if self.rng.random() < p["tcell_recruit_rate"] * dt:
                free2 = self.grid.free_neighbors(vxi, vyi)
                if free2:
                    nx, ny = free2[self.rng.integers(len(free2))]
                    tc = TCell(nx, ny, p, self.rng)
                    self.tcells.append(tc)
                    self.grid.occupancy[ny, nx] = -(id(tc))

            # MDSC recruitment
            if self.rng.random() < p["mdsc_recruit_rate"] * dt:
                free3 = self.grid.free_neighbors(vxi, vyi)
                if free3:
                    nx, ny = free3[self.rng.integers(len(free3))]
                    mdsc = MDSCCell(nx, ny, p, self.rng)
                    self.mdscs.append(mdsc)
                    self.grid.occupancy[ny, nx] = -(id(mdsc))

            # Treg recruitment
            if self.rng.random() < p["treg_recruit_rate"] * dt:
                free4 = self.grid.free_neighbors(vxi, vyi)
                if free4:
                    nx, ny = free4[self.rng.integers(len(free4))]
                    treg = TregCell(nx, ny, p, self.rng)
                    self.tregs.append(treg)
                    self.grid.occupancy[ny, nx] = -(id(treg))

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self):
        dt = SIM["dt"]
        p  = TUMOR_CELL
        ip = IMMUNE

        all_immune = self.tams + self.tcells + self.mdscs + self.tregs

        # 1. O2 and glucose consumption maps
        o2_consume = self.o2_solver.build_consumption_map(
            self.grid, self.cells,
            {"GSC":       OXYGEN["consume_quiesc"],
             "PROLIF":    OXYGEN["consume_prolif"],
             "INVASIVE":  OXYGEN["consume_prolif"] * 0.7,
             "QUIESCENT": OXYGEN["consume_quiesc"],
             "NECROTIC":  0.0}
        )
        glc_consume = self.glc_solver.build_consumption_map(
            self.grid, self.cells,
            {"PROLIF":    GLUCOSE["consume_prolif"],
             "INVASIVE":  GLUCOSE["consume_prolif"],
             "GSC":       GLUCOSE["consume_prolif"] * 0.5,
             "QUIESCENT": GLUCOSE["consume_quiesc"],
             "NECROTIC":  0.0}
        )

        # 2. Advance substrate fields
        self.o2_solver.step(self.grid.oxygen, self.grid.vessels,
                            OXYGEN["vessel_conc"], o2_consume)
        self.glc_solver.step(self.grid.glucose, self.grid.vessels,
                             GLUCOSE["vessel_conc"], glc_consume)

        # 3. Advance signaling fields (now includes immune secretion)
        self.signaling.step(self.grid, self.cells,
                            immune_cells=all_immune,
                            immune_params=ip)

        # 4. Angiogenesis
        self._maybe_sprout_vessels()

        # 5. Recruit new immune cells through vessels
        self._recruit_immune_cells()

        # 6. Build suppression map from MDSCs and Tregs
        suppression_map = self._build_suppression_map()

        # 7. Build tumor cell lookup dict (id -> cell) for immune killing
        tumor_by_id = {cell.id: cell for cell in self.cells}

        # 8. Update tumor cells
        self.rng.shuffle(self.cells)
        new_cells = []
        dead_tumor_ids = set()

        for cell in self.cells:
            local_o2  = self.grid.oxygen[cell.y, cell.x]
            local_glc = self.grid.glucose[cell.y, cell.x]

            cell.update_state(local_o2, local_glc, dt, p)

            if not cell.alive:
                dead_tumor_ids.add(cell.id)
                continue

            daughter = cell.try_divide(self.grid, dt, p)
            if daughter:
                new_cells.append(daughter)

            cell.try_migrate(self.grid, self.grid.oxygen, dt, p)

        # 9. Update TAMs — polarisation + migration + killing
        dead_from_tam = set()
        for tam in self.tams:
            tam.update(self.grid, self.signaling, dt, ip)
            if tam.alive:
                killed = tam.try_kill(self.grid, tumor_by_id, dt, ip)
                if killed:
                    dead_from_tam.add(killed)

        # 10. Update T cells — exhaustion + migration + killing
        dead_from_tcell = set()
        for tc in self.tcells:
            tc.update(self.grid, self.signaling, dt, ip)
            if tc.alive:
                local_supp = suppression_map.get((tc.x, tc.y), 1.0)
                killed = tc.try_kill(self.grid, tumor_by_id, dt, ip,
                                     local_suppression=local_supp)
                if killed:
                    dead_from_tcell.add(killed)

        # 11. Update MDSCs and Tregs
        for mdsc in self.mdscs:
            mdsc.update(self.grid, self.signaling, dt, ip)
        for treg in self.tregs:
            treg.update(self.grid, self.signaling, dt, ip)

        # 12. Collect all tumor cells killed by immune system
        immune_killed = dead_from_tam | dead_from_tcell

        # 13. Remove dead tumor cells
        all_dead = dead_tumor_ids | immune_killed
        for cell in self.cells:
            if cell.id in all_dead:
                self.grid.occupancy[cell.y, cell.x] = -1

        self.cells = [c for c in self.cells if c.id not in all_dead]
        self.cells.extend(new_cells)
        # Add new daughters to lookup
        for d in new_cells:
            tumor_by_id[d.id] = d

        # 14. Remove dead immune cells
        def _clean(cell_list):
            dead = [c for c in cell_list if not c.alive]
            for c in dead:
                if self.grid.in_bounds(c.x, c.y):
                    if self.grid.occupancy[c.y, c.x] == -(id(c)):
                        self.grid.occupancy[c.y, c.x] = -1
            return [c for c in cell_list if c.alive]

        self.tams   = _clean(self.tams)
        self.tcells = _clean(self.tcells)
        self.mdscs  = _clean(self.mdscs)
        self.tregs  = _clean(self.tregs)

    # ------------------------------------------------------------------
    # Suppression map
    # ------------------------------------------------------------------

    def _build_suppression_map(self):
        """
        Returns dict: (x,y) -> suppression_factor [0,1].
        1.0 = no suppression (T cell kills at full rate).
        Values below 1.0 reduce T cell killing.

        MDSCs suppress within radius mdsc_suppress_radius.
        Tregs suppress within radius treg_suppress_radius.
        Multiple suppressors stack multiplicatively.
        """
        ip = IMMUNE
        supp = {}   # (x,y) -> factor

        for mdsc in self.mdscs:
            zone = mdsc.get_suppression_zone(self.grid, ip)
            for pos in zone:
                supp[pos] = supp.get(pos, 1.0) * ip["mdsc_suppress_factor"]

        for treg in self.tregs:
            zone = treg.get_suppression_zone(self.grid, ip)
            for pos in zone:
                supp[pos] = supp.get(pos, 1.0) * ip["treg_suppress_factor"]

        return supp

    # ------------------------------------------------------------------
    # Angiogenesis (unchanged from Month 3)
    # ------------------------------------------------------------------

    def _maybe_sprout_vessels(self):
        max_vessels   = 80
        current_count = int(self.grid.vessels.sum())
        if current_count >= max_vessels:
            return

        candidates  = self.signaling.get_angiogenesis_stimulus(self.grid)
        sprout_prob = 0.000005
        vy, vx = np.where(self.grid.vessels)

        for (cx, cy) in candidates:
            if current_count >= max_vessels:
                break
            if self.grid.vessels[cy, cx]:
                continue
            if len(vx) == 0:
                continue
            min_dist = np.sqrt(((vx - cx)**2 + (vy - cy)**2)).min()
            if min_dist > 10:
                continue
            if self.rng.random() < sprout_prob:
                self.grid.vessels[cy, cx] = True
                current_count += 1

    # ------------------------------------------------------------------
    # Run and snapshot
    # ------------------------------------------------------------------

    def run(self):
        t = 0.0
        n_steps       = int(SIM["total_time"] / SIM["dt"])
        save_interval = int(SIM["save_every"] / SIM["dt"])

        for step_i in tqdm(range(n_steps), desc="Simulating"):
            self.step()
            t += SIM["dt"]

            if step_i % save_interval == 0:
                self._record_snapshot(t)

        return self.history

    def _record_snapshot(self, t):
        # Tumor state counts
        tumor_states = {}
        for cell in self.cells:
            s = cell.state.name
            tumor_states[s] = tumor_states.get(s, 0) + 1

        # Immune counts
        m1_count = sum(1 for c in self.tams if c.state.name == "M1")
        m2_count = sum(1 for c in self.tams if c.state.name == "M2")
        active_t = sum(1 for c in self.tcells if c.state.name == "ACTIVE")
        exh_t    = sum(1 for c in self.tcells if c.state.name == "EXHAUSTED")

        sig = self.signaling.get_summary()

        self.history.append({
            "time_hr":       t,
            "n_cells":       len(self.cells),
            "tumor_states":  tumor_states,
            "o2_mean":       float(self.grid.oxygen.mean()),
            "o2_min":        float(self.grid.oxygen.min()),
            "vessel_count":  int(self.grid.vessels.sum()),
            # Immune
            "n_tam_m1":      m1_count,
            "n_tam_m2":      m2_count,
            "n_tcell_active":active_t,
            "n_tcell_exh":   exh_t,
            "n_mdsc":        len(self.mdscs),
            "n_treg":        len(self.tregs),
            # Signaling
            "vegf_mean":     sig["vegf_mean"],
            "vegf_max":      sig["vegf_max"],
            "tgf_mean":      sig["tgf_mean"],
            "tgf_max":       sig["tgf_max"],
            "il10_mean":     sig["il10_mean"],
            "ifng_mean":     sig["ifng_mean"],
            "ifng_max":      sig["ifng_max"],
        })