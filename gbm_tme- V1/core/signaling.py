import numpy as np
from core.diffusion import DiffusionSolver


class SignalingLayer:
    """
    Month 3 — Molecular signaling layer.

    Manages three diffusible fields:
      1. VEGF     — secreted by hypoxic/invasive cells, drives angiogenesis
      2. TGF_beta — secreted by tumor cells, immunosuppressive (Month 4)
      3. IL10     — secreted by tumor cells, second immunosuppressive cytokine

    PDE:  dC/dt = D * nabla2(C) - decay * C + secretion(cells)
    """

    def __init__(self, grid_params, signal_params, dt):
        W = grid_params["width"]
        H = grid_params["height"]

        self.vegf     = np.zeros((H, W))
        self.tgf_beta = np.zeros((H, W))
        self.il10     = np.zeros((H, W))

        self.p  = signal_params
        self.dt = dt

        self.vegf_solver = DiffusionSolver(
            D=signal_params["vegf_D"], dx=grid_params["voxel_size"], dt=dt)
        self.tgf_solver = DiffusionSolver(
            D=signal_params["tgf_D"],  dx=grid_params["voxel_size"], dt=dt)
        self.il10_solver = DiffusionSolver(
            D=signal_params["il10_D"], dx=grid_params["voxel_size"], dt=dt)

        print("SignalingLayer initialized — VEGF, TGF-b, IL-10 fields ready.")

    def build_secretion_maps(self, grid, cells):
        H, W = self.vegf.shape
        vegf_sec     = np.zeros((H, W))
        tgf_beta_sec = np.zeros((H, W))
        il10_sec     = np.zeros((H, W))
        p = self.p

        for cell in cells:
            y, x = cell.y, cell.x
            name = cell.state.name

            if name == "INVASIVE":
                vegf_sec[y, x]     += p["vegf_sec_invasive"]
                tgf_beta_sec[y, x] += p["tgf_sec_tumor"]
                il10_sec[y, x]     += p["il10_sec_tumor"]

            elif name == "PROLIF":
                vegf_sec[y, x]     += p["vegf_sec_prolif"]
                tgf_beta_sec[y, x] += p["tgf_sec_tumor"]
                il10_sec[y, x]     += p["il10_sec_tumor"]

            elif name == "GSC":
                vegf_sec[y, x]     += p["vegf_sec_prolif"]
                tgf_beta_sec[y, x] += p["tgf_sec_gsc"]
                il10_sec[y, x]     += p["il10_sec_tumor"]

            elif name == "NECROTIC":
                vegf_sec[y, x]     += p["vegf_sec_necrotic"]

        return vegf_sec, tgf_beta_sec, il10_sec

    def step(self, grid, cells):
        vegf_sec, tgf_sec, il10_sec = self.build_secretion_maps(grid, cells)
        p = self.p

        self._step_field(self.vegf,     self.vegf_solver,
                         vegf_sec,  p["vegf_decay"],  p["vegf_max"])
        self._step_field(self.tgf_beta, self.tgf_solver,
                         tgf_sec,   p["tgf_decay"],   p["tgf_max"])
        self._step_field(self.il10,     self.il10_solver,
                         il10_sec,  p["il10_decay"],  p["il10_max"])

    def _step_field(self, field, solver, secretion, decay, max_conc):
        empty_mask = np.zeros(field.shape, dtype=bool)
        solver.step(
            field           = field,
            source_mask     = empty_mask,
            source_conc     = 0.0,
            consumption_map = -secretion,   # negative = acts as source
            decay           = decay
        )
        np.clip(field, 0.0, max_conc, out=field)

    def get_angiogenesis_stimulus(self, grid):
        threshold = self.p["vegf_sprout_threshold"]
        vy, vx = np.where(self.vegf > threshold)
        return list(zip(vx.tolist(), vy.tolist()))

    def get_summary(self):
        return {
            "vegf_mean":   float(self.vegf.mean()),
            "vegf_max":    float(self.vegf.max()),
            "tgf_mean":    float(self.tgf_beta.mean()),
            "tgf_max":     float(self.tgf_beta.max()),
            "il10_mean":   float(self.il10.mean()),
            "il10_max":    float(self.il10.max()),
            "angio_sites": int((self.vegf > self.p["vegf_sprout_threshold"]).sum()),
        }