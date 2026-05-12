# config/params.py
# Units: time=hours, space=micrometers (um),
#        O2=mmHg, glucose=mM, cytokines=pg/mL
#        TMZ concentration=uM, radiation dose=Gy

GRID = {
    "width":      200,
    "height":     200,
    "voxel_size": 20,
}

OXYGEN = {
    "D":               1800.0,
    "decay":           0.0,
    "vessel_conc":     40.0,
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
    "vegf_D":                180.0,
    "vegf_decay":            0.1,
    "vegf_sec_invasive":     2.0,
    "vegf_sec_prolif":       0.3,
    "vegf_sec_necrotic":     0.1,
    "vegf_max":              200.0,
    "vegf_sprout_threshold": 2.0,
    "tgf_D":                 120.0,
    "tgf_decay":             0.15,
    "tgf_sec_tumor":         0.5,
    "tgf_sec_gsc":           1.0,
    "tgf_max":               100.0,
    "il10_D":                200.0,
    "il10_decay":            0.2,
    "il10_sec_tumor":        0.3,
    "il10_max":              80.0,
    "ifng_D":                220.0,
    "ifng_decay":            0.25,
    "ifng_sec_m1tam":        0.4,
    "ifng_sec_tcell":        0.6,
    "ifng_max":              60.0,
}

IMMUNE = {
    "tam_recruit_rate":      0.005,
    "tcell_recruit_rate":    0.002,
    "mdsc_recruit_rate":     0.002,
    "treg_recruit_rate":     0.001,
    "tam_migration_speed":   4.0,
    "tam_lifespan":          240.0,
    "tgf_m2_threshold":      0.3,
    "il10_m2_threshold":     0.2,
    "ifng_m1_threshold":     0.2,
    "m1_to_m2_prob":         0.3,
    "m2_to_m1_prob":         0.05,
    "m1_kill_prob":          0.02,
    "m1_tgf_secretion":      0.0,
    "m1_il10_secretion":     0.0,
    "m1_ifng_secretion":     0.4,
    "m2_kill_prob":          0.0,
    "m2_tgf_secretion":      0.3,
    "m2_il10_secretion":     0.2,
    "m2_ifng_secretion":     0.0,
    "tcell_migration_speed": 6.0,
    "tcell_lifespan":        120.0,
    "tcell_kill_prob":       0.08,
    "tcell_kill_radius":     1,
    "exhaustion_tgf_thresh": 0.4,
    "exhaustion_il10_thresh":0.3,
    "exhaustion_rate":       0.05,
    "exhaustion_threshold":  1.0,
    "exhaustion_kill_factor":0.1,
    "tcell_ifng_secretion":  0.6,
    "mdsc_migration_speed":  3.0,
    "mdsc_lifespan":         96.0,
    "mdsc_suppress_radius":  3,
    "mdsc_suppress_factor":  0.3,
    "treg_migration_speed":  5.0,
    "treg_lifespan":         144.0,
    "treg_suppress_radius":  2,
    "treg_suppress_factor":  0.4,
    "treg_il10_secretion":   0.15,
    "treg_tgf_secretion":    0.1,
}

# ------------------------------------------------------------------
# Month 5 — Treatment parameters (Stupp Protocol)
# ------------------------------------------------------------------
# Reference: Stupp et al. 2005 (NEJM) — standard GBM treatment
# TMZ PK: Portnow et al. 2009; Ostermann et al. 2004
# MGMT: Hegi et al. 2005 (NEJM)
# Radiobiology: Hall & Giaccia 2019

TREATMENT = {
    # ── Simulation duration ────────────────────────────────────────
    # Extended to 42 days to cover full concurrent Stupp phase
    "total_time_hr":        1008.0,   # 42 days in hours

    # ── TMZ pharmacokinetics ───────────────────────────────────────
    # TMZ diffuses through tissue from vessel sources
    # BBB reduces plasma→CNS concentration to ~20%
    "tmz_D":                400.0,    # um2/hr — small molecule, faster diffusion
    "tmz_decay":            0.139,    # /hr — plasma half-life ~1.8 hr → decay 0.139
    "tmz_bbb_fraction":     0.20,     # 20% of plasma concentration crosses BBB
    "tmz_plasma_conc":      50.0,     # uM — peak plasma concentration at 75mg/m2/day
    "tmz_vessel_conc":      10.0,     # uM — effective CNS concentration (20% of plasma)
    "tmz_max":              50.0,     # uM — hard cap

    # TMZ DNA damage — probability of lethal damage per cell per hour
    # Scales with local TMZ concentration
    # Reference: Portnow et al. 2009
    "tmz_kill_base":        0.002,    # /hr/uM — base killing rate per uM TMZ

    # MGMT methylation
    # Switch: "all_methylated", "all_unmethylated", "mixed"
    # mixed: mgmt_methylation_rate fraction of cells are methylated
    "mgmt_scenario":        "mixed",  # ← change this to test scenarios
    "mgmt_methylation_rate":0.40,     # 40% methylated — clinical GBM rate
                                      # (Hegi et al. 2005: ~45% in trial)

    # MGMT repair — unmethylated cells repair TMZ damage
    # Methylated cells CANNOT repair (MGMT promoter silenced)
    "mgmt_repair_factor":   0.05,     # unmethylated: only 5% of damage is lethal
                                      # methylated: 100% of damage is lethal

    # GSCs are ALWAYS unmethylated — treatment resistant
    # This drives recurrence after treatment ends
    "gsc_tmz_resistance":   0.005,    # GSCs: only 0.5% killing rate vs bulk

    # ── Radiotherapy (RT) ──────────────────────────────────────────
    # Standard: 2 Gy per fraction, 5 days/week, 6 weeks = 30 fractions
    # Concurrent with TMZ days 1-42
    "rt_dose_per_fraction": 2.0,      # Gy per fraction
    "rt_fractions_per_week":5,        # Monday-Friday
    "rt_start_day":         1,        # Day 1 of simulation
    "rt_end_day":           42,       # Day 42 (30 fractions over 6 weeks)

    # Linear-quadratic model: survival fraction = exp(-alpha*D - beta*D²)
    # Parameters for GBM (Joiner & van der Kogel 2009)
    "rt_alpha":             0.3,      # Gy⁻¹
    "rt_beta":              0.03,     # Gy⁻²
    "rt_alpha_beta":        10.0,     # Gy (alpha/beta ratio for GBM)

    # Oxygen Enhancement Ratio (OER)
    # Hypoxic cells are ~3x more radioresistant
    # OER reduces effective dose in hypoxic voxels
    # Reference: Hall & Giaccia 2019 (Radiobiology for the Radiologist)
    "rt_oer_max":           3.0,      # maximum OER (fully anoxic)
    "rt_oer_k":             3.0,      # mmHg — O2 concentration for half-max OER
                                      # (OER = OER_max * K / (pO2 + K))

    # GSC radioresistance — GSCs activate DNA damage checkpoints
    # Reference: Bao et al. 2006 (Nature)
    "rt_gsc_resistance":    0.2,      # GSC survival fraction raised to power 0.2
                                      # (higher = more resistant)

    # ── Stupp protocol schedule ────────────────────────────────────
    # Phase 1: Concurrent RT + TMZ (days 1-42) — modelled here
    # Phase 2: Rest (days 43-70) — optional extension
    # Phase 3: Adjuvant TMZ (days 71-210, 6 cycles) — future work
    "concurrent_start_day": 1,
    "concurrent_end_day":   42,

    # Treatment ON/OFF switch — set False to run control (no treatment)
    "treatment_active":     True,
}

SIM = {
    "dt":              0.05,
    "total_time":      TREATMENT["total_time_hr"],
    "n_initial_cells": 50,
    "n_initial_gsc":   5,            # explicitly seeded GSCs at start
    "seed":            42,
    "save_every":      24,
}