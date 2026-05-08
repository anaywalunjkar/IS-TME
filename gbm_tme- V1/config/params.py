# config/params.py
# All parameters in one file.
# Units: time in hours, space in micrometers (um),
#        O2 concentration in mmHg, glucose in mM

GRID = {
    "width":      200,     # voxels
    "height":     200,
    "voxel_size": 20,      # um per voxel (~cell diameter)
}

OXYGEN = {
    "D":              1800.0,  # um2/hr — diffusion coefficient in brain tissue
    "decay":          0.0,
    "vessel_conc":    40.0,    # mmHg — pO2 at vessel wall (normoxic)
    "consume_prolif": 2.0,     # mmHg/hr per cell — proliferating consumption
    "consume_quiesc": 0.5,     # mmHg/hr per cell — quiescent/GSC consumption
    "hypoxia_thresh": 8.0,     # mmHg — below this = hypoxic phenotype
    "necrosis_thresh":1.0,     # mmHg — below this = necrosis
}

GLUCOSE = {
    "D":              600.0,   # um2/hr
    "vessel_conc":    5.0,     # mM
    "consume_prolif": 0.2,     # mM/hr per cell (Warburg effect)
    "consume_quiesc": 0.05,    # mM/hr per cell
    "starvation_thresh": 0.5,  # mM — below this = starving
}

TUMOR_CELL = {
    # GSC (glioblastoma stem cell)
    "gsc_cycle_time":        48.0,   # hr — slower cycling than bulk
    "gsc_symmetric_prob":    0.1,    # prob of symmetric GSC->GSC+GSC division
    "symmetric_prob":        0.1,    # alias used in try_divide

    # Proliferating bulk cells
    "prolif_cycle_time":     24.0,   # hr
    "prolif_death_rate":     0.01,   # per hr baseline apoptosis

    # Migration
    "migration_speed":       10.0,   # um/hr
    "chemotaxis_strength":   0.5,    # bias toward O2 gradient (0=random, 1=pure chemotaxis)

    # State transition probabilities (per hour — scaled by dt in code)
    "hypoxia_to_invasive_prob":    2.0,   # /hr under hypoxia -> invasive
    "starvation_to_necrosis_time": 48,    # hr without glucose -> necrosis
    "invasive_revert_prob": 0.005,   # was 0.05 — much slower reversion

    # Microenvironment thresholds (duplicated from OXYGEN/GLUCOSE for cell access)
    "hypoxia_thresh":   8.0,    # mmHg
    "necrosis_thresh":  1.0,    # mmHg
    "starvation_thresh":0.5,    # mM
}

SIM = {
    "dt":             0.05,    # hr — timestep (must satisfy stability: dt < dx2/(4D))
    "total_time":     720.0,   # hr = 30 days
    "n_initial_cells":50,      # seed tumor cells at start
    "seed":           42,      # random seed for reproducibility
    "save_every":     24,      # hr — snapshot interval
}