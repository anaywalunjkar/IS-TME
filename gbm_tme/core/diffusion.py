import numpy as np
from scipy.ndimage import laplace

class DiffusionSolver:
    """Solves the reaction-diffusion PDE for each substrate.

    PDE:  dC/dt = D * ∇²C - consumption(cells) + source(vessels)

    Uses explicit finite difference — simple but requires small dt.
    Stability condition: dt < dx² / (4 * D)
    Check this at init or your simulation will blow up silently.
    """

    def __init__(self, D, dx, dt):
        self.D = D      # diffusion coefficient µm²/hr
        self.dx = dx    # voxel size µm
        self.dt = dt    # timestep hr

        # Stability check — this is critical
        max_stable_dt = (dx ** 2) / (4.0 * D)
        if dt > max_stable_dt:
            raise ValueError(
                f"Unstable: dt={dt} > max stable dt={max_stable_dt:.4f}. "
                f"Reduce dt or increase dx."
            )

    def step(self, field, source_mask, source_conc,
             consumption_map, decay=0.0):
        """
        Advance field by one timestep.

        Args:
            field:           2D concentration array (modified in place)
            source_mask:     bool array — True at vessel voxels
            source_conc:     concentration at source (Dirichlet BC)
            consumption_map: 2D array of consumption rates per voxel
            decay:           first-order decay constant (1/hr)

        Returns:
            Updated field (same array, modified in place)
        """
        # Laplacian via scipy (handles boundary as zero-flux by default)
        lap = laplace(field) / (self.dx ** 2)

        # Update
        field += self.dt * (
            self.D * lap
            - consumption_map
            - decay * field
        )

        # Apply vessel source (Dirichlet — clamp vessels to source_conc)
        field[source_mask] = source_conc

        # Physical constraint — no negative concentrations
        np.clip(field, 0.0, None, out=field)

        return field

    def build_consumption_map(self, grid, cells, consume_rates):
        """
        Build a 2D consumption array from cell positions and their
        consumption rates keyed by cell state.

        consume_rates: dict like {"proliferating": 6.25, "gsc": 1.5, ...}
        """
        cmap = np.zeros((grid.H, grid.W))
        for cell in cells:
            rate = consume_rates.get(cell.state.name, 0.0)
            cmap[cell.y, cell.x] += rate
        return cmap