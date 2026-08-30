"""
Central configuration. Every downstream module reads from here rather
than hardcoding values, so horizon/window/alpha stay consistent across
label construction, windowing, conformal calibration and evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"

PARQUET = PROCESSED_DIR / "alibaba_ssd_processed.parquet"

SMART_DIRS = [RAW_DIR / "smartlog2018ssd", RAW_DIR / "smartlog2019ssd"]
FAILURE_GLOB = str(RAW_DIR / "ssd_failure_label.csv" / "**" / "*.csv")


# ---------------------------------------------------------------
# Schema
# ---------------------------------------------------------------

META_COLS = ["disk_id", "ds", "model"]
DERIVED_COLS = ["failure_time", "days_to_failure", "failure_event"]
NON_SMART_COLS = set(META_COLS) | set(DERIVED_COLS)

DRIVE_KEY = ["model", "disk_id"]

# Vendor is the 2nd character of the model code: MA1 -> A.
# verify_and_profile.py raises if any model code breaks this.
VENDOR_SLICE = (1, 1)

DRIVE_MODELS = ["MA1", "MA2", "MB1", "MB2", "MC1", "MC2"]
VENDORS = ["A", "B", "C"]


@dataclass(frozen=True)
class Config:
    # -- Label construction -------------------------------------
    horizon_days: int = 30          # WEFR's prediction horizon
    drop_post_failure: bool = True  # rows with days_to_failure < 0
    drop_censored: bool = False     # drives ending before global end

    # -- Windowing ----------------------------------------------
    window_len: int = 30            # daily observations per sequence
    stride: int = 1
    min_valid_frac: float = 0.5     # min non-missing fraction to keep

    # -- Conformal ----------------------------------------------
    alpha: float = 0.10             # target miscoverage -> 90% coverage
    cal_frac: float = 0.20          # of training vendors' drives

    # -- Splits -------------------------------------------------
    test_frac: float = 0.20         # standard (non-LOMO) split only
    seed: int = 42

    # -- WEFR ---------------------------------------------------
    n_rankers: int = 10
    kendall_outlier_z: float = 2.0

    # -- Synthetic fixtures -------------------------------------
    # Calibrated to reports/vendor_profile.csv. Per-model failure rates
    # live in data/synthetic.py; failure_scale multiplies them so small
    # fixtures carry enough positives while preserving the real A<B<C
    # ordering. censor_rate matches the real 133,804/475,056 = 28%.
    synth_n_drives: int = 400
    synth_days: int = 180
    synth_failure_scale: float = 4.0
    synth_censor_rate: float = 0.28
    synth_seed: int = 42

    # -- LOMO ---------------------------------------------------
    lomo_axis: str = "vendor"       # "vendor" (3 folds) | "model" (6)
    folds: tuple = field(default_factory=lambda: tuple(VENDORS))


CFG = Config()