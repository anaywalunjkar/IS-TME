import numpy as np
from tqdm import tqdm
from core.grid import Grid
from core.tumor_cell import TumorCell, CellState
from core.diffusion import DiffusionSolver
from config.params import GRID, OXYGEN, GLUCOSE, TUMOR_CELL, SIM
 
 
class SimulationEngine:
 
    def __init__(self):
        self.rng = np.random.default_rng(SIM["seed"])
        self.grid = Grid(GRID)
        self.grid.initialize_vessels(n_vessels=40, rng=self.rng) # increase to 40
        self.grid.initialize_substrates(OXYGEN, GLUCOSE)
 
        self.o2_solver  = DiffusionSolver(OXYGEN["D"],  GRID["voxel_size"], SIM["dt"])
        self.glc_solver = DiffusionSolver(GLUCOSE["D"], GRID["voxel_size"], SIM["dt"])
 
        self.cells = []
        self.history = []
 
        print("Pre-equilibrating substrate fields...")
        empty_consume = np.zeros((GRID["height"], GRID["width"]))
        for _ in range(2000):
            self.o2_solver.step(self.grid.oxygen, self.grid.vessels,
                                OXYGEN["vessel_conc"], empty_consume)
            self.glc_solver.step(self.grid.glucose, self.grid.vessels,
                                 GLUCOSE["vessel_conc"], empty_consume)
        print(f"O2 range after equilibration: "
              f"{self.grid.oxygen.min():.1f} - {self.grid.oxygen.max():.1f} mmHg")
 
        self._seed_tumor()
 
    def _seed_tumor(self):
        vy, vx = np.where(self.grid.vessels)
        cx, cy = GRID["width"] // 2, GRID["height"] // 2
 
        dists = ((vx - cx)**2 + (vy - cy)**2)
        nearest = np.argmin(dists)
        sx, sy = int(vx[nearest]), int(vy[nearest])
 
        placed = 0
        attempts = 0
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
 
    def step(self):
        dt = SIM["dt"]
        p  = TUMOR_CELL
 
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
 
        self.o2_solver.step(
            self.grid.oxygen, self.grid.vessels,
            OXYGEN["vessel_conc"], o2_consume
        )
        self.glc_solver.step(
            self.grid.glucose, self.grid.vessels,
            GLUCOSE["vessel_conc"], glc_consume
        )
 
        self.rng.shuffle(self.cells)
        new_cells = []
        dead_ids  = set()
 
        for cell in self.cells:
            local_o2  = self.grid.oxygen[cell.y, cell.x]
            local_glc = self.grid.glucose[cell.y, cell.x]
 
            cell.update_state(local_o2, local_glc, dt, p)
 
            if not cell.alive:
                dead_ids.add(cell.id)
                continue
 
            daughter = cell.try_divide(self.grid, dt, p)
            if daughter:
                new_cells.append(daughter)
 
            cell.try_migrate(self.grid, self.grid.oxygen, dt, p)
 
        for cell in self.cells:
            if cell.id in dead_ids:
                self.grid.occupancy[cell.y, cell.x] = -1
 
        self.cells = [c for c in self.cells if c.id not in dead_ids]
        self.cells.extend(new_cells)
 
    def run(self):
        t = 0.0
        n_steps = int(SIM["total_time"] / SIM["dt"])
        save_interval = int(SIM["save_every"] / SIM["dt"])
 
        for step_i in tqdm(range(n_steps), desc="Simulating"):
            self.step()
            t += SIM["dt"]
 
            if step_i % save_interval == 0:
                self._record_snapshot(t)
 
        return self.history
 
    def _record_snapshot(self, t):
        state_counts = {}
        for cell in self.cells:
            s = cell.state.name
            state_counts[s] = state_counts.get(s, 0) + 1
 
        self.history.append({
            "time_hr":  t,
            "n_cells":  len(self.cells),
            "states":   state_counts,
            "o2_mean":  float(self.grid.oxygen.mean()),
            "o2_min":   float(self.grid.oxygen.min()),
        })