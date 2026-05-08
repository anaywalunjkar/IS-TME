import numpy as np

class Grid:
    """2D spatial lattice. Each voxel tracks substrate concentrations
    and which cell (if any) occupies it."""

    def __init__(self, params):
        self.W = params["width"]
        self.H = params["height"]
        self.dx = params["voxel_size"]  # µm

        # Substrate fields — shape (H, W)
        self.oxygen  = np.zeros((self.H, self.W))
        self.glucose = np.zeros((self.H, self.W))

        # Occupancy: cell_id or None per voxel
        self.occupancy = np.full((self.H, self.W), -1, dtype=int)

        # Vessel mask — True where a vessel is present
        self.vessels = np.zeros((self.H, self.W), dtype=bool)

    def initialize_vessels(self, n_vessels=20, rng=None):
        """Scatter random vessel points — simple approximation of
        brain vasculature. Each vessel supplies O₂ and glucose."""
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(n_vessels):
            x = rng.integers(10, self.W - 10)
            y = rng.integers(10, self.H - 10)
            self.vessels[y, x] = True

    def initialize_substrates(self, o2_params, glc_params):
        """Set initial concentrations. Vessels = source, rest = 0."""
        self.oxygen[self.vessels]  = o2_params["vessel_conc"]
        self.glucose[self.vessels] = glc_params["vessel_conc"]

    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def get_neighbors(self, x, y, moore=True):
        """Return valid neighbor (x,y) coordinates."""
        if moore:
            offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        else:
            offsets = [(-1,0),(1,0),(0,-1),(0,1)]
        return [(x+dx, y+dy) for dx,dy in offsets
                if self.in_bounds(x+dx, y+dy)]

    def free_neighbors(self, x, y):
        return [(nx,ny) for nx,ny in self.get_neighbors(x,y)
                if self.occupancy[ny, nx] == -1]