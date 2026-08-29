"""
Tests for the synthetic fixture generator.

These assert the properties downstream code depends on. If one fails,
the fixture is no longer a valid stand-in for the real parquet and any
test built on it is meaningless.

    pytest tests/test_synthetic.py -v
"""

import numpy as np
import polars as pl
import pytest

from src.config import CFG, DRIVE_MODELS
from src.schema import ALL_COLS, SMART_COLS, vendor_cols
from src.data.synthetic import make_synthetic


@pytest.fixture(scope="module")
def fixture():
    return make_synthetic()


# ---------------------------------------------------------------
# Schema
# ---------------------------------------------------------------

def test_column_count(fixture):
    """105 schema columns + 3 derived from the failure join."""
    df, _ = fixture
    assert len(ALL_COLS) == 105
    expected = ALL_COLS + ["failure_time", "days_to_failure",
                           "failure_event"]
    assert df.columns == expected


def test_dtypes(fixture):
    df, _ = fixture
    assert df.schema["disk_id"] == pl.Int64
    assert df.schema["ds"] == pl.Date
    assert df.schema["model"] == pl.String
    for c in SMART_COLS:
        assert df.schema[c] == pl.Float64, c


def test_all_models_present(fixture):
    df, _ = fixture
    assert set(df["model"].unique()) == set(DRIVE_MODELS)


# ---------------------------------------------------------------
# The NaN-vs-null trap
# ---------------------------------------------------------------

def test_missing_encoded_as_nan_not_null(fixture):
    """
    preprocess_data.py used pd.to_numeric(errors='coerce'), producing
    NaN. Polars treats NaN and null as distinct. The fixture must
    reproduce this or availability checks pass for the wrong reason.
    """
    df, _ = fixture
    a = df.filter(pl.col("model").str.starts_with("MA"))
    b_only = [c for c in vendor_cols("B") if c not in vendor_cols("A")]
    assert b_only, "vendor availability map has no divergence"

    col = b_only[0]
    assert a[col].is_nan().all(), f"{col} should be all-NaN for vendor A"
    assert not a[col].is_null().any(), f"{col} used null, not NaN"


def test_vendor_availability_diverges(fixture):
    """The covariate shift the paper measures must exist in the fixture."""
    df, truth = fixture
    sets = {v: set(ids) for v, ids in truth["vendor_smart_ids"].items()}
    assert sets["A"] != sets["B"]
    assert sets["B"] != sets["C"]
    assert sets["A"] != sets["C"]


def test_vendor_columns_populated(fixture):
    df, _ = fixture
    for vendor, prefix in [("A", "MA"), ("B", "MB"), ("C", "MC")]:
        sub = df.filter(pl.col("model").str.starts_with(prefix))
        for c in vendor_cols(vendor):
            frac = float((~sub[c].is_nan()).mean())
            assert frac > 0.9, f"{c} sparse for vendor {vendor}: {frac}"


# ---------------------------------------------------------------
# Drive key
# ---------------------------------------------------------------

def test_drive_key_is_model_plus_disk_id(fixture):
    """disk_id alone must not identify a drive."""
    df, _ = fixture
    pairs = df.select(["model", "disk_id"]).unique()
    collisions = (
        pairs.group_by("disk_id").len().filter(pl.col("len") > 1).height
    )
    assert collisions > 0, "fixture must exercise the (model, disk_id) key"


def test_one_row_per_drive_day(fixture):
    df, _ = fixture
    assert df.select(["model", "disk_id", "ds"]).unique().height == df.height


# ---------------------------------------------------------------
# Labels
# ---------------------------------------------------------------

def test_failure_count_matches_truth(fixture):
    df, truth = fixture
    observed = (
        df.filter(pl.col("failure_time").is_not_null())
        .select(["model", "disk_id"]).unique().height
    )
    assert observed == truth["n_failed"]


def test_horizon_label_window_size(fixture):
    """
    Each failed drive must have exactly horizon_days rows in
    0 < days_to_failure <= horizon_days. This is the assertion that
    validates labels.py once it exists.
    """
    df, truth = fixture
    h = truth["horizon_days"]
    counts = (
        df.filter(
            (pl.col("days_to_failure") > 0)
            & (pl.col("days_to_failure") <= h)
        )
        .group_by(["model", "disk_id"]).len()
    )
    assert counts.height == truth["n_failed"]
    assert counts["len"].min() == h
    assert counts["len"].max() == h


def test_no_rows_after_failure(fixture):
    df, _ = fixture
    assert df.filter(pl.col("days_to_failure") < 0).height == 0


def test_failure_event_is_not_the_target(fixture):
    """
    failure_event is a 'has already failed' flag: one row per failed
    drive. Guards against anyone training on it by mistake.
    """
    df, truth = fixture
    assert int(df["failure_event"].sum()) == truth["n_failed"]


def test_healthy_drives_have_null_failure_time(fixture):
    df, truth = fixture
    failed = {(m, d) for m, d, _ in truth["failed"]}
    healthy = df.filter(pl.col("failure_time").is_null())
    for m, d in healthy.select(["model", "disk_id"]).unique().iter_rows():
        assert (m, d) not in failed


# ---------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------

def test_censored_drives_end_early(fixture):
    df, truth = fixture
    last = (
        df.group_by(["model", "disk_id"])
        .agg(pl.col("ds").max().alias("last_ds"))
    )
    end = truth["global_end"]
    for m, d, expected_last in truth["censored"]:
        row = last.filter(
            (pl.col("model") == m) & (pl.col("disk_id") == d)
        )
        assert row["last_ds"][0] == expected_last
        assert row["last_ds"][0] < end


def test_censored_drives_are_not_failures(fixture):
    _, truth = fixture
    failed = {(m, d) for m, d, _ in truth["failed"]}
    censored = {(m, d) for m, d, _ in truth["censored"]}
    assert not (failed & censored)


# ---------------------------------------------------------------
# Injected signal
# ---------------------------------------------------------------

def test_signal_present_in_designated_columns(fixture):
    """
    Raw values in SIGNAL_IDS must be higher in the pre-failure horizon
    than earlier in the drive's life. WEFR feature selection should
    recover these IDs; that test lives in tests/test_wefr.py.
    """
    df, truth = fixture
    h = truth["horizon_days"]

    for sid in truth["signal_ids"]:
        col = f"r_{sid}"
        sub = df.filter(pl.col("failure_time").is_not_null())
        near = sub.filter(
            (pl.col("days_to_failure") > 0)
            & (pl.col("days_to_failure") <= h)
        )[col]
        far = sub.filter(pl.col("days_to_failure") > 2 * h)[col]

        near_m = float(np.nanmean(near.to_numpy()))
        far_m = float(np.nanmean(far.to_numpy()))
        assert near_m > far_m, f"no signal in {col}: {near_m} vs {far_m}"


def test_no_signal_in_healthy_drives(fixture):
    """Healthy drives must not carry the drift, or the task is trivial."""
    df, truth = fixture
    sid = truth["signal_ids"][0]
    col = f"r_{sid}"

    healthy = df.filter(pl.col("failure_time").is_null())
    tail = healthy.group_by(["model", "disk_id"]).tail(truth["horizon_days"])
    head = healthy.group_by(["model", "disk_id"]).head(truth["horizon_days"])

    jump = (float(np.nanmean(tail[col].to_numpy()))
            - float(np.nanmean(head[col].to_numpy())))
    assert jump < 300, "healthy drives show failure-like drift"


# ---------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------

def test_seed_is_deterministic():
    a, ta = make_synthetic(seed=7)
    b, tb = make_synthetic(seed=7)
    assert a.equals(b)
    assert ta["failed"] == tb["failed"]


def test_different_seeds_differ():
    a, _ = make_synthetic(seed=1)
    b, _ = make_synthetic(seed=2)
    assert not a.equals(b)


# ---------------------------------------------------------------
# Scale knobs
# ---------------------------------------------------------------

def test_small_fixture_is_usable():
    """Tiny fixtures must stay valid — used for fast unit tests."""
    df, truth = make_synthetic(n_drives=40, n_days=90, seed=3)
    assert truth["n_failed"] >= 3
    assert df.height > 0
    assert set(df["model"].unique()) == set(DRIVE_MODELS)


def test_vendor_c_is_thin(fixture):
    """
    Vendor C is deliberately under-represented so thin-fold handling is
    exercised on every run, mirroring the real LOMO viability risk.
    """
    df, _ = fixture
    per_vendor = (
        df.with_columns(pl.col("model").str.slice(1, 1).alias("vendor"))
        .group_by("vendor")
        .agg(pl.col("disk_id").n_unique().alias("n"))
        .sort("n")
    )
    assert per_vendor["vendor"][0] == "C"
