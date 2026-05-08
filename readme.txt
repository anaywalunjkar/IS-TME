# GBM Tumor Microenvironment — In Silico Simulation

A multiscale computational simulation of the Glioblastoma (GBM) tumor microenvironment built in Python, integrating agent-based modelling (ABM) with partial differential equation (PDE) solvers to reproduce the complex biology of high-grade gliomas in silico.

---

## What this simulates

Glioblastoma Multiforme (GBM, WHO Grade IV) is the most aggressive primary brain tumor in adults. This simulation models the tumor as an evolving ecosystem of interacting cell agents and diffusible molecular fields over a 30-day virtual timeline.

### Biological components modelled

| Layer | What is modelled |
|---|---|
| **Tumor cells** | GSC, Proliferating, Invasive, Necrotic — with HIF-1α driven state transitions |
| **Oxygen** | Diffusion from vessels, consumption by cells, hypoxia gradients |
| **Glucose** | Warburg effect — aerobic glycolysis by proliferating cells |
| **VEGF** | Secreted by hypoxic cells, drives angiogenesis |
| **TGF-β** | Immunosuppressive cytokine — drives TAM M1→M2 polarisation |
| **IL-10** | Second immunosuppressive cytokine — suppresses T cell activity |
| **IFN-γ** | Anti-tumor cytokine — secreted by M1 TAMs and active T cells |
| **TAMs** | M1/M2 polarising Tumor-Associated Macrophages |
| **CD8+ T cells** | Cytotoxic T cells with exhaustion dynamics |
| **MDSCs** | Myeloid-Derived Suppressor Cells — suppress T cell killing |
| **Tregs** | Regulatory T cells — suppress T cell activity |

---

## Architecture

The simulation uses a **hybrid PDE-ABM approach** inspired by Frieboes et al. 2007:
- **PDE layer**: finite-difference diffusion solvers for 6 substrate/signaling fields
- **Agent layer**: discrete cell objects with state machines, division, and migration
- **Coupling**: cells read local field concentrations; cells update field secretion maps each timestep

```
gbm_tme/
├── config/params.py          # All biological parameters (single source of truth)
├── core/
│   ├── grid.py               # 2D spatial lattice (200x200 voxels, 20 µm each)
│   ├── cell.py               # BaseCell parent class
│   ├── tumor_cell.py         # GBM cell states, division, migration
│   ├── immune_cell.py        # TAM, TCell, MDSC, Treg agents
│   ├── diffusion.py          # Explicit finite-difference PDE solver
│   └── signaling.py          # VEGF, TGF-β, IL-10, IFN-γ fields
├── simulation/engine.py      # Master loop — couples all modules per timestep
├── visualization/plotter.py  # Matplotlib output figures
├── tests/
│   ├── test_diffusion.py     # PDE solver unit tests
│   ├── test_cells.py         # Tumor cell transition tests
│   └── test_immune.py        # Immune cell behaviour tests
└── run_sim.py                # Entry point
```

---

## Validated biological benchmarks

| Benchmark | Clinical value | Simulation | Status |
|---|---|---|---|
| Necrotic fraction | 30–40% of tumor | ~35–40% by day 15–20 | ✅ |
| Three-ring morphology | Necrotic core → hypoxic rim → proliferating edge | Visible in spatial maps | ✅ |
| Invasive cell escape | GBM invades beyond main mass | Invasive cells escape boundary | ✅ |
| VEGF at hypoxic rim | HIF-1α drives VEGF at tumor-brain interface | VEGF peak at invasive rim | ✅ |
| TAM M2 dominance | >70% M2 TAMs in GBM TME | 70–87% M2 fraction at day 30 | ✅ |
| T cell exhaustion | T cells exhausted in GBM core | Exhaustion accumulates under TGF-β | ✅ |
| Immunosuppression rising | TGF-β and IL-10 scale with tumor burden | Both rise monotonically | ✅ |

---

## Sample outputs

The simulation produces 6 figures per run:

- `output_spatial.png` — O₂ field, tumor cell states, glucose field at day 30
- `output_immune.png` — TAM M1/M2 positions, T cell states, TGF-β vs IFN-γ overlay
- `output_signaling.png` — VEGF, TGF-β, IL-10, IFN-γ spatial fields at day 30
- `output_growth.png` — Tumor growth curve and cell state dynamics over 30 days
- `output_immune_timeseries.png` — TAM polarisation, T cell exhaustion, cytokine balance
- `output_signaling_timeseries.png` — VEGF + angiogenesis, TGF-β, IL-10 over time

---

## Installation and usage

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gbm-tme-simulation.git
cd gbm-tme-simulation

# Install dependencies
pip install numpy scipy matplotlib pandas tqdm jupyter

# Run unit tests first
python tests/test_diffusion.py
python tests/test_cells.py
python tests/test_immune.py

# Run the full 30-day simulation (~15-25 min on CPU)
python run_sim.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, PDE fields |
| `scipy` | Laplacian computation (ndimage.laplace) |
| `matplotlib` | All visualisation |
| `tqdm` | Progress bar |
| `pandas` | Snapshot history analysis |

Python 3.10+ required.

---

## Key parameters

All biological constants are in `config/params.py`. Every value has a literature source comment. Key parameters:

| Parameter | Value | Source |
|---|---|---|
| O₂ diffusion coefficient | 1800 µm²/hr | Grote et al. 1977 |
| Hypoxia threshold | 8.0 mmHg | Semenza 2001 |
| GBM proliferation cycle | 24 hr | Stensjoen et al. 2015 |
| GSC cycle time | 48 hr | Lathia et al. 2015 |
| VEGF half-life | ~7 hr | Ferrara et al. 2003 |
| TAM M1→M2 probability | 0.3/hr under TGF-β | Hambardzumyan et al. 2016 |

---

## Roadmap

- [x] Month 1–2: Core tumor model — cell states, O₂/glucose PDE, necrotic core
- [x] Month 3: Signaling layer — VEGF, TGF-β, IL-10, angiogenesis
- [x] Month 4: Immune compartment — TAMs, T cells, MDSCs, Tregs, IFN-γ
- [ ] Month 5: Treatment module — TMZ PK/PD, radiation, MGMT methylation
- [ ] Month 6: 3D extension, patient-specific imaging integration

---

## References

Key papers this simulation is grounded in:

1. Neftel et al. (2019) — GBM cell state map. *Cell* 178(4)
2. Frieboes et al. (2007) — Hybrid PDE-ABM GBM model. *NeuroImage* 37:S59
3. Quail & Joyce (2017) — GBM tumor microenvironment. *Cancer Cell* 31(3)
4. Hambardzumyan et al. (2016) — TAM biology in GBM. *Nature Neuroscience* 19(2)
5. Semenza (2001) — HIF-1α and hypoxia signalling. *Cell* 107(1)
6. Ferrara et al. (2003) — VEGF biology. *Nature Medicine* 9(6)
7. Woroniecka et al. (2018) — T cell exhaustion in GBM. *Clinical Cancer Research* 24(17)

Full reference list with DOIs available in the repository documentation.

---

## Author

Built as an independent computational oncology research project.
Simulation architecture, biological parameterisation, and validation by [Your Name].

---

*This is an active research project. Month 5 (treatment module) in development.*