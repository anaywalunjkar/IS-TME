import numpy as np
from tqdm import tqdm
from core.grid import Grid
from core.tumor_cell import TumorCell, CellState
from core.diffusion import DiffusionSolver
from core.signaling import SignalingLayer
from core.immune_cell import TAMCell, TCell, MDSCCell, TregCell, TAMState
from core.treatment import TreatmentModule
from config.params import (GRID, OXYGEN, GLUCOSE, TUMOR_CELL,
                           SIGNALING, IMMUNE, TREATMENT, SIM)


class SimulationEngine:

    def __init__(self):
        self.rng = np.random.default_rng(SIM["seed"])
        self.grid = Grid(GRID)
        self.grid.initialize_vessels(n_vessels=40, rng=self.rng)
        self.grid.initialize_substrates(OXYGEN, GLUCOSE)

        self.o2_solver  = DiffusionSolver(OXYGEN["D"],  GRID["voxel_size"], SIM["dt"])
        self.glc_solver = DiffusionSolver(GLUCOSE["D"], GRID["voxel_size"], SIM["dt"])
        self.signaling  = SignalingLayer(GRID, SIGNALING, SIM["dt"])

        # Month 5 — treatment module
        self.treatment = TreatmentModule(GRID, TREATMENT, SIM["dt"])

        self.cells  = []
        self.tams   = []
        self.tcells = []
        self.mdscs  = []
        self.tregs  = []
        self.history = []

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

        scenario = TREATMENT["mgmt_scenario"]
        rate     = TREATMENT["mgmt_methylation_rate"]

        placed, attempts = 0, 0
        while placed < SIM["n_initial_cells"] and attempts < 1000:
            x = sx + self.rng.integers(-8, 9)
            y = sy + self.rng.integers(-8, 9)
            if (self.grid.in_bounds(x, y) and
                    self.grid.occupancy[y, x] == -1 and
                    self.grid.oxygen[y, x] > OXYGEN["hypoxia_thresh"]):

                # Assign MGMT status based on scenario
                if scenario == "all_methylated":
                    mgmt = True
                elif scenario == "all_unmethylated":
                    mgmt = False
                else:   # mixed
                    mgmt = bool(self.rng.random() < rate)

                cell = TumorCell(x, y, CellState.PROLIF, TUMOR_CELL,
                                 self.rng, mgmt_methylated=mgmt)
                self.cells.append(cell)
                self.grid.occupancy[y, x] = cell.id
                placed += 1
            attempts += 1

        # Count methylation at seeding
        n_meth = sum(1 for c in self.cells if c.mgmt_methylated)
        print(f"Seeded {placed} tumor cells near vessel at ({sx}, {sy})")
        print(f"  MGMT scenario: {scenario}")
        print(f"  Methylated: {n_meth} / {placed} "
              f"({n_meth/placed*100:.0f}%)")

        # Explicitly seed GSC subpopulation — always unmethylated, treatment resistant
        n_gsc = SIM.get("n_initial_gsc", 5)
        gsc_placed, gsc_attempts = 0, 0
        while gsc_placed < n_gsc and gsc_attempts < 500:
            x = sx + self.rng.integers(-6, 7)
            y = sy + self.rng.integers(-6, 7)
            if (self.grid.in_bounds(x, y) and
                    self.grid.occupancy[y, x] == -1 and
                    self.grid.oxygen[y, x] > OXYGEN["hypoxia_thresh"]):
                gsc = TumorCell(x, y, CellState.GSC, TUMOR_CELL,
                                self.rng, mgmt_methylated=False)
                self.cells.append(gsc)
                self.grid.occupancy[y, x] = gsc.id
                gsc_placed += 1
            gsc_attempts += 1
        print(f"  GSC seeds placed: {gsc_placed}")

    def _recruit_immune_cells(self):
        vy, vx = np.where(self.grid.vessels)
        p  = IMMUNE
        dt = SIM["dt"]

        for (vxi, vyi) in zip(vx.tolist(), vy.tolist()):
            free = self.grid.free_neighbors(vxi, vyi)
            if not free:
                continue

            if self.rng.random() < p["tam_recruit_rate"] * dt:
                nx, ny = free[self.rng.integers(len(free))]
                tam = TAMCell(nx, ny, TAMState.M1, p, self.rng)
                self.tams.append(tam)
                self.grid.occupancy[ny, nx] = -(id(tam))

            if self.rng.random() < p["tcell_recruit_rate"] * dt:
                free2 = self.grid.free_neighbors(vxi, vyi)
                if free2:
                    nx, ny = free2[self.rng.integers(len(free2))]
                    tc = TCell(nx, ny, p, self.rng)
                    self.tcells.append(tc)
                    self.grid.occupancy[ny, nx] = -(id(tc))

            if self.rng.random() < p["mdsc_recruit_rate"] * dt:
                free3 = self.grid.free_neighbors(vxi, vyi)
                if free3:
                    nx, ny = free3[self.rng.integers(len(free3))]
                    mdsc = MDSCCell(nx, ny, p, self.rng)
                    self.mdscs.append(mdsc)
                    self.grid.occupancy[ny, nx] = -(id(mdsc))

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

        # 1. Advance treatment clock
        self.treatment.advance(dt)

        # 2. O2 and glucose consumption
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

        # 3. Advance substrate fields
        self.o2_solver.step(self.grid.oxygen, self.grid.vessels,
                            OXYGEN["vessel_conc"], o2_consume)
        self.glc_solver.step(self.grid.glucose, self.grid.vessels,
                             GLUCOSE["vessel_conc"], glc_consume)

        # 4. Advance TMZ field
        self.treatment.step_tmz(self.grid)

        # 5. Advance signaling fields
        self.signaling.step(self.grid, self.cells,
                            immune_cells=all_immune,
                            immune_params=ip)

        # 6. Angiogenesis
        self._maybe_sprout_vessels()

        # 7. Recruit immune cells
        self._recruit_immune_cells()

        # 8. Suppression map
        suppression_map = self._build_suppression_map()

        # 9. Tumor cell lookup
        tumor_by_id = {cell.id: cell for cell in self.cells}

        # 10. Apply TMZ damage to all tumor cells
        self.treatment.apply_tmz_to_cells(self.cells, dt)

        # 11. Apply radiation if today is a treatment day
        if self.treatment.rt_today:
            self.treatment.apply_radiation_to_cells(
                self.cells, self.grid.oxygen)

        # 12. Update tumor cells
        self.rng.shuffle(self.cells)
        new_cells  = []
        dead_tumor = set()

        for cell in self.cells:
            local_o2  = self.grid.oxygen[cell.y, cell.x]
            local_glc = self.grid.glucose[cell.y, cell.x]

            cell.update_state(local_o2, local_glc, dt, p)

            if not cell.alive:
                dead_tumor.add(cell.id)
                continue

            daughter = cell.try_divide(self.grid, dt, p)
            if daughter:
                new_cells.append(daughter)

            cell.try_migrate(self.grid, self.grid.oxygen, dt, p)

        # 13. Update immune cells
        dead_from_tam   = set()
        dead_from_tcell = set()

        for tam in self.tams:
            tam.update(self.grid, self.signaling, dt, ip)
            if tam.alive:
                killed = tam.try_kill(self.grid, tumor_by_id, dt, ip)
                if killed:
                    dead_from_tam.add(killed)

        for tc in self.tcells:
            tc.update(self.grid, self.signaling, dt, ip)
            if tc.alive:
                local_supp = suppression_map.get((tc.x, tc.y), 1.0)
                killed = tc.try_kill(self.grid, tumor_by_id, dt, ip,
                                     local_suppression=local_supp)
                if killed:
                    dead_from_tcell.add(killed)

        for mdsc in self.mdscs:
            mdsc.update(self.grid, self.signaling, dt, ip)
        for treg in self.tregs:
            treg.update(self.grid, self.signaling, dt, ip)

        # 14. Remove dead tumor cells
        all_dead = dead_tumor | dead_from_tam | dead_from_tcell
        for cell in self.cells:
            if cell.id in all_dead:
                self.grid.occupancy[cell.y, cell.x] = -1

        self.cells = [c for c in self.cells if c.id not in all_dead]
        self.cells.extend(new_cells)

        # 15. Remove dead immune cells
        def _clean(lst):
            dead = [c for c in lst if not c.alive]
            for c in dead:
                if self.grid.in_bounds(c.x, c.y):
                    if self.grid.occupancy[c.y, c.x] == -(id(c)):
                        self.grid.occupancy[c.y, c.x] = -1
            return [c for c in lst if c.alive]

        self.tams   = _clean(self.tams)
        self.tcells = _clean(self.tcells)
        self.mdscs  = _clean(self.mdscs)
        self.tregs  = _clean(self.tregs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_suppression_map(self):
        ip   = IMMUNE
        supp = {}
        for mdsc in self.mdscs:
            for pos in mdsc.get_suppression_zone(self.grid, ip):
                supp[pos] = supp.get(pos, 1.0) * ip["mdsc_suppress_factor"]
        for treg in self.tregs:
            for pos in treg.get_suppression_zone(self.grid, ip):
                supp[pos] = supp.get(pos, 1.0) * ip["treg_suppress_factor"]
        return supp

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
        tumor_states = {}
        for cell in self.cells:
            s = cell.state.name
            tumor_states[s] = tumor_states.get(s, 0) + 1

        m1 = sum(1 for c in self.tams   if c.state.name == "M1")
        m2 = sum(1 for c in self.tams   if c.state.name == "M2")
        ta = sum(1 for c in self.tcells if c.state.name == "ACTIVE")
        te = sum(1 for c in self.tcells if c.state.name == "EXHAUSTED")

        sig   = self.signaling.get_summary()
        treat = self.treatment.get_summary()
        mgmt  = self.treatment.get_mgmt_counts(self.cells)

        self.history.append({
            "time_hr":        t,
            "day":            t / 24.0,
            "n_cells":        len(self.cells),
            "tumor_states":   tumor_states,
            "o2_mean":        float(self.grid.oxygen.mean()),
            "o2_min":         float(self.grid.oxygen.min()),
            "vessel_count":   int(self.grid.vessels.sum()),
            # Immune
            "n_tam_m1":       m1,
            "n_tam_m2":       m2,
            "n_tcell_active": ta,
            "n_tcell_exh":    te,
            "n_mdsc":         len(self.mdscs),
            "n_treg":         len(self.tregs),
            # Signaling
            "vegf_mean":      sig["vegf_mean"],
            "tgf_mean":       sig["tgf_mean"],
            "il10_mean":      sig["il10_mean"],
            "ifng_mean":      sig["ifng_mean"],
            # Treatment
            "tmz_active":     treat["tmz_active"],
            "rt_fractions":   treat["fraction_count"],
            "tmz_mean":       treat["tmz_mean"],
            "tmz_max":        treat["tmz_max"],
            # MGMT
            "n_methylated":   mgmt["methylated"],
            "n_unmethylated": mgmt["unmethylated"],
            "n_gsc":          mgmt["gsc"],
        })