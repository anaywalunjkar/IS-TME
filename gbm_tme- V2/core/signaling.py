import numpy as np
from core.diffusion import DiffusionSolver


class SignalingLayer:
    """
    Molecular signaling layer.

    Month 3 fields: VEGF, TGF-beta, IL-10
    Month 4 addition: IFN-gamma

    IFN-gamma is secreted by M1 TAMs and active CD8+ T cells.
    It drives M2 -> M1 TAM reversion (rare in GBM but critical for
    immunotherapy response modelling in Month 5+).
    """

    def __init__(self, grid_params, signal_params, dt):
        W = grid_params["width"]
        H = grid_params["height"]

        # Month 3 fields
        self.vegf     = np.zeros((H, W))
        self.tgf_beta = np.zeros((H, W))
        self.il10     = np.zeros((H, W))

        # Month 4 addition
        self.ifng     = np.zeros((H, W))

        self.p  = signal_params
        self.dt = dt

        self.vegf_solver = DiffusionSolver(
            D=signal_params["vegf_D"], dx=grid_params["voxel_size"], dt=dt)
        self.tgf_solver = DiffusionSolver(
            D=signal_params["tgf_D"],  dx=grid_params["voxel_size"], dt=dt)
        self.il10_solver = DiffusionSolver(
            D=signal_params["il10_D"], dx=grid_params["voxel_size"], dt=dt)
        self.ifng_solver = DiffusionSolver(
            D=signal_params["ifng_D"], dx=grid_params["voxel_size"], dt=dt)

        print("SignalingLayer initialized — VEGF, TGF-b, IL-10, IFN-g fields ready.")

    # ------------------------------------------------------------------
    # Tumor cell secretion maps (Month 3 — unchanged)
    # ------------------------------------------------------------------

    def build_tumor_secretion_maps(self, grid, tumor_cells):
        H, W = self.vegf.shape
        vegf_sec = np.zeros((H, W))
        tgf_sec  = np.zeros((H, W))
        il10_sec = np.zeros((H, W))
        p = self.p

        for cell in tumor_cells:
            y, x = cell.y, cell.x
            name = cell.state.name

            if name == "INVASIVE":
                vegf_sec[y, x] += p["vegf_sec_invasive"]
                tgf_sec[y, x]  += p["tgf_sec_tumor"]
                il10_sec[y, x] += p["il10_sec_tumor"]
            elif name == "PROLIF":
                vegf_sec[y, x] += p["vegf_sec_prolif"]
                tgf_sec[y, x]  += p["tgf_sec_tumor"]
                il10_sec[y, x] += p["il10_sec_tumor"]
            elif name == "GSC":
                vegf_sec[y, x] += p["vegf_sec_prolif"]
                tgf_sec[y, x]  += p["tgf_sec_gsc"]
                il10_sec[y, x] += p["il10_sec_tumor"]
            elif name == "NECROTIC":
                vegf_sec[y, x] += p["vegf_sec_necrotic"]

        return vegf_sec, tgf_sec, il10_sec

    # ------------------------------------------------------------------
    # Immune cell secretion maps (Month 4 — new)
    # ------------------------------------------------------------------

    def build_immune_secretion_maps(self, grid, immune_cells, immune_params):
        """
        Build secretion maps from immune cells.

        TAM M1: secretes IFN-g
        TAM M2: secretes TGF-b, IL-10
        T cell (active): secretes IFN-g
        Treg: secretes TGF-b, IL-10

        Returns: (tgf_sec, il10_sec, ifng_sec) arrays
        """
        H, W = self.vegf.shape
        tgf_sec  = np.zeros((H, W))
        il10_sec = np.zeros((H, W))
        ifng_sec = np.zeros((H, W))

        p  = immune_params

        for cell in immune_cells:
            y, x = cell.y, cell.x
            ctype = cell.cell_type

            if ctype == "TAM":
                t, i, g = cell.get_secretion(p)
                tgf_sec[y, x]  += t
                il10_sec[y, x] += i
                ifng_sec[y, x] += g

            elif ctype == "TCELL":
                ifng_sec[y, x] += cell.get_ifng_secretion(p)

            elif ctype == "TREG":
                t, i = cell.get_secretion(p)
                tgf_sec[y, x]  += t
                il10_sec[y, x] += i

            # MDSCs do not secrete cytokines directly

        return tgf_sec, il10_sec, ifng_sec

    # ------------------------------------------------------------------
    # Master step
    # ------------------------------------------------------------------

    def step(self, grid, tumor_cells, immune_cells=None, immune_params=None):
        """
        Advance all four fields one timestep.

        Secretion = tumor cells (Month 3) + immune cells (Month 4).
        """
        # Tumor secretion
        vegf_sec, tgf_tumor, il10_tumor = \
            self.build_tumor_secretion_maps(grid, tumor_cells)

        # Immune secretion (Month 4)
        tgf_immune  = np.zeros_like(self.tgf_beta)
        il10_immune = np.zeros_like(self.il10)
        ifng_sec    = np.zeros_like(self.ifng)

        if immune_cells and immune_params:
            tgf_immune, il10_immune, ifng_sec = \
                self.build_immune_secretion_maps(
                    grid, immune_cells, immune_params)

        # Combined secretion
        tgf_total  = tgf_tumor  + tgf_immune
        il10_total = il10_tumor + il10_immune

        p = self.p

        # Advance fields
        self._step_field(self.vegf,     self.vegf_solver,
                         vegf_sec,   p["vegf_decay"],  p["vegf_max"])
        self._step_field(self.tgf_beta, self.tgf_solver,
                         tgf_total,  p["tgf_decay"],   p["tgf_max"])
        self._step_field(self.il10,     self.il10_solver,
                         il10_total, p["il10_decay"],  p["il10_max"])
        self._step_field(self.ifng,     self.ifng_solver,
                         ifng_sec,   p["ifng_decay"],  p["ifng_max"])

    def _step_field(self, field, solver, secretion, decay, max_conc):
        empty_mask = np.zeros(field.shape, dtype=bool)
        solver.step(
            field           = field,
            source_mask     = empty_mask,
            source_conc     = 0.0,
            consumption_map = -secretion,
            decay           = decay
        )
        np.clip(field, 0.0, max_conc, out=field)

    # ------------------------------------------------------------------
    # Angiogenesis trigger
    # ------------------------------------------------------------------

    def get_angiogenesis_stimulus(self, grid):
        threshold = self.p["vegf_sprout_threshold"]
        vy, vx = np.where(self.vegf > threshold)
        return list(zip(vx.tolist(), vy.tolist()))

    # ------------------------------------------------------------------
    # Summary readout
    # ------------------------------------------------------------------

    def get_summary(self):
        return {
            "vegf_mean":   float(self.vegf.mean()),
            "vegf_max":    float(self.vegf.max()),
            "tgf_mean":    float(self.tgf_beta.mean()),
            "tgf_max":     float(self.tgf_beta.max()),
            "il10_mean":   float(self.il10.mean()),
            "il10_max":    float(self.il10.max()),
            "ifng_mean":   float(self.ifng.mean()),
            "ifng_max":    float(self.ifng.max()),
            "angio_sites": int((self.vegf > self.p["vegf_sprout_threshold"]).sum()),
        }