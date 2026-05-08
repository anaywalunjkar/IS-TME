def __init__(self):
    self.rng = np.random.default_rng(SIM["seed"])
    self.grid = Grid(GRID)
    self.grid.initialize_vessels(n_vessels=40, rng=self.rng)
    self.grid.initialize_substrates(OXYGEN, GLUCOSE)

    self.o2_solver  = DiffusionSolver(OXYGEN["D"],  GRID["voxel_size"], SIM["dt"])
    self.glc_solver = DiffusionSolver(GLUCOSE["D"], GRID["voxel_size"], SIM["dt"])

    self.cells = []
    self.history = []

    # ↓ ADD THIS — run diffusion 2000 steps before placing any cells
    # so O₂ and glucose are spatially distributed when cells appear
    print("Pre-equilibrating substrate fields...")
    empty_consume = __import__('numpy').zeros((GRID["height"], GRID["width"]))
    for _ in range(2000):
        self.o2_solver.step(self.grid.oxygen, self.grid.vessels,
                            OXYGEN["vessel_conc"], empty_consume)
        self.glc_solver.step(self.grid.glucose, self.grid.vessels,
                             GLUCOSE["vessel_conc"], empty_consume)
    print(f"O2 range after equilibration: "
          f"{self.grid.oxygen.min():.1f} – {self.grid.oxygen.max():.1f} mmHg")

    self._seed_tumor()