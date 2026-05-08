# config/params.py
# Units: time=hours, space=micrometers (um),
#        O2=mmHg, glucose=mM, cytokines=pg/mL-equivalent

GRID = {
    "width":      200,
    "height":     200,
    "voxel_size": 20,      # um per voxel
}

OXYGEN = {
    "D":               1800.0,
    "decay":           0.0,
    "vessel_conc":     40.0,    # mmHg
    "consume_prolif":  2.0,
    "consume_quiesc":  0.5,
    "hypoxia_thresh":  8.0,
    "necrosis_thresh": 1.0,
}

GLUCOSE = {
    "D":                 600.0,
    "vessel_conc":       5.0,
    "consume_prolif":    0.2,
    "consume_quiesc":    0.05,
    "starvation_thresh": 0.5,
}

TUMOR_CELL = {
    "gsc_cycle_time":              48.0,
    "gsc_symmetric_prob":          0.1,
    "symmetric_prob":              0.1,
    "prolif_cycle_time":           24.0,
    "prolif_death_rate":           0.01,
    "migration_speed":             10.0,
    "chemotaxis_strength":         0.5,
    "hypoxia_to_invasive_prob":    2.0,
    "starvation_to_necrosis_time": 48,
    "invasive_revert_prob":        0.001,
    "hypoxia_thresh":              8.0,
    "necrosis_thresh":             1.0,
    "starvation_thresh":           0.5,
}

SIGNALING = {
    # VEGF
    "vegf_D":                180.0,
    "vegf_decay":            0.1,
    "vegf_sec_invasive":     2.0,
    "vegf_sec_prolif":       0.3,
    "vegf_sec_necrotic":     0.1,
    "vegf_max":              200.0,
    "vegf_sprout_threshold": 2.0,
    # TGF-beta
    "tgf_D":                 120.0,
    "tgf_decay":             0.15,
    "tgf_sec_tumor":         0.5,
    "tgf_sec_gsc":           1.0,
    "tgf_max":               100.0,
    # IL-10
    "il10_D":                200.0,
    "il10_decay":            0.2,
    "il10_sec_tumor":        0.3,
    "il10_max":              80.0,
    # IFN-gamma (Month 4 — secreted by M1 TAMs and active T cells)
    "ifng_D":                220.0,   # um2/hr — small cytokine, fast diffusion
    "ifng_decay":            0.25,    # /hr — half-life ~2.8 hr (degrades quickly)
    "ifng_sec_m1tam":        0.4,     # pg/mL/hr — M1 TAMs are strong IFN-g source
    "ifng_sec_tcell":        0.6,     # pg/mL/hr — active CD8+ T cells
    "ifng_max":              60.0,    # pg/mL
}

# ------------------------------------------------------------------
# Month 4 — Immune compartment parameters
# ------------------------------------------------------------------
IMMUNE = {
    # ── Recruitment / extravasation ────────────────────────────────
    # Immune cells enter through vessels each timestep with low prob
    # Rates calibrated so TAM:T cell ratio ~5:1 at steady state
    # (matches clinical GBM data: TAMs dominate infiltrate)
    "tam_recruit_rate":   0.005,   # ← was 0.002, 2.5x higher
    "tcell_recruit_rate": 0.002,   # ← was 0.0005, 4x higher
    "mdsc_recruit_rate":  0.002,   # ← was 0.0008
    "treg_recruit_rate":  0.001,   # ← was 0.0003

    # ── TAM (Tumor-Associated Macrophage) ──────────────────────────
    # M1 = pro-inflammatory, anti-tumor
    # M2 = anti-inflammatory, pro-tumor (dominant in GBM)
    "tam_migration_speed":  4.0,     # um/hr — slower than tumor cells
    "tam_lifespan":         240.0,   # hr = 10 days in tissue

    # Polarisation thresholds (pg/mL)
    # Above tgf_m2_threshold: TAM polarises toward M2
    "tgf_m2_threshold":     0.3,     # pg/mL TGF-b drives M1->M2
    "il10_m2_threshold":    0.2,     # pg/mL IL-10 reinforces M2
    "ifng_m1_threshold":    0.2,     # pg/mL IFN-g drives M2->M1

    # Polarisation probabilities per hour
    "m1_to_m2_prob":        0.3,     # /hr under TGF-b + IL-10
    "m2_to_m1_prob":        0.05,    # /hr under IFN-g (rare — hard to reverse)

    # M1 TAM actions
    "m1_kill_prob":         0.02,    # /hr prob of killing adjacent tumor cell
    "m1_tgf_secretion":     0.0,     # M1 does NOT secrete TGF-b
    "m1_il10_secretion":    0.0,
    "m1_ifng_secretion":    0.4,     # secreted by M1 per hr

    # M2 TAM actions
    "m2_kill_prob":         0.0,     # M2 does not kill tumor cells
    "m2_tgf_secretion":     0.3,     # reinforces immunosuppression
    "m2_il10_secretion":    0.2,
    "m2_ifng_secretion":    0.0,

    # ── CD8+ T cell ────────────────────────────────────────────────
    "tcell_migration_speed": 6.0,    # um/hr — faster than TAMs
    "tcell_lifespan":        120.0,  # hr = 5 days before natural death

    # Killing
    "tcell_kill_prob":       0.08,   # /hr base killing rate when adjacent
    "tcell_kill_radius":     1,      # voxels — must be adjacent (Moore neighbourhood)

    # Exhaustion — driven by local TGF-b and IL-10
    "exhaustion_tgf_thresh": 0.4,    # pg/mL — above this, exhaustion accumulates
    "exhaustion_il10_thresh":0.3,    # pg/mL
    "exhaustion_rate":       0.05,   # /hr — rate of exhaustion accumulation
    "exhaustion_threshold":  1.0,    # cumulative — above this = EXHAUSTED state
    "exhaustion_kill_factor":0.1,    # exhausted T cells kill at 10% of normal rate

    # T cell IFN-g secretion
    "tcell_ifng_secretion":  0.6,    # pg/mL/hr when active

    # ── MDSC (Myeloid-Derived Suppressor Cell) ─────────────────────
    "mdsc_migration_speed":  3.0,    # um/hr — slow
    "mdsc_lifespan":         96.0,   # hr = 4 days
    "mdsc_suppress_radius":  3,      # voxels — suppresses T cells within this radius
    "mdsc_suppress_factor":  0.3,    # T cell killing rate multiplied by this

    # ── Treg (Regulatory T cell) ───────────────────────────────────
    "treg_migration_speed":  5.0,    # um/hr
    "treg_lifespan":         144.0,  # hr = 6 days
    "treg_suppress_radius":  2,      # voxels
    "treg_suppress_factor":  0.4,    # T cell killing suppressed to 40%
    "treg_il10_secretion":   0.15,   # pg/mL/hr
    "treg_tgf_secretion":    0.1,    # pg/mL/hr
}

SIM = {
    "dt":              0.05,
    "total_time":      720.0,   # hr = 30 days
    "n_initial_cells": 50,
    "seed":            42,
    "save_every":      24,
}