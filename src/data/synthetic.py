"""
Synthetic fixture generator.

Produces a small dataframe matching the real Alibaba parquet schema, with
properties chosen in advance so downstream logic can be asserted against
known answers rather than eyeballed.

Reproduced deliberately:
  - per-vendor attribute availability differences (the covariate shift)
  - missingness encoded as NaN, not null (pd.to_numeric(errors='coerce'))
  - right-censored drives that stop reporting before the global end
  - monotonic drift in designated columns before failure (the signal)

Usage:
    df, truth = make_synthetic()
    df                      # pl.DataFrame, same columns as the parquet
    truth["failed"]         # [(model, disk_id, failure_date), ...]
    truth["signal_ids"]     # SMART IDs carrying the pre-failure signal
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from ..config import CFG, DRIVE_MODELS
from ..schema import ALL_COLS, SMART_COLS, VENDOR_SMART_IDS, vendor_cols

START = date(2018, 1, 1)

# SMART IDs into which pre-failure drift is injected. Chosen from the
# common set so every vendor carries signal. WEFR feature selection
# should recover these; that is the test.
SIGNAL_IDS = [5, 187, 197]

# Uneven drive allocation across models. Vendor C is deliberately thin,
# so thin-fold handling gets exercised on every test run.
MODEL_WEIGHTS = {
    "MA1": 0.26, "MA2": 0.20,
    "MB1": 0.22, "MB2": 0.16,
    "MC1": 0.10, "MC2": 0.06,
}


def _vendor_of(model: str) -> str:
    return model[1]


def make_synthetic(
    n_drives: int | None = None,
    n_days: int | None = None,
    failure_rate: float | None = None,
    censor_rate: float | None = None,
    seed: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Build a synthetic dataset plus its ground truth."""
    n_drives = n_drives if n_drives is not None else CFG.synth_n_drives
    n_days = n_days if n_days is not None else CFG.synth_days
    failure_rate = (failure_rate if failure_rate is not None
                    else CFG.synth_failure_rate)
    censor_rate = (censor_rate if censor_rate is not None
                   else CFG.synth_censor_rate)
    seed = seed if seed is not None else CFG.synth_seed

    rng = np.random.default_rng(seed)
    global_end = START + timedelta(days=n_days - 1)

    # -- allocate drives to models ------------------------------
    models, counts = [], []
    for m in DRIVE_MODELS:
        k = max(2, int(round(n_drives * MODEL_WEIGHTS[m])))
        models.append(m)
        counts.append(k)

    drives = []
    next_id = {m: 1000 for m in DRIVE_MODELS}
    for m, k in zip(models, counts):
        for _ in range(k):
            drives.append((m, next_id[m]))
            next_id[m] += 1

    # -- assign failures and censoring --------------------------
    n_fail = max(3, int(round(len(drives) * failure_rate)))
    fail_idx = set(rng.choice(len(drives), size=n_fail, replace=False)
                   .tolist())

    remaining = [i for i in range(len(drives)) if i not in fail_idx]
    n_cens = int(round(len(drives) * censor_rate))
    cens_idx = set(rng.choice(remaining,
                              size=min(n_cens, len(remaining)),
                              replace=False).tolist())

    failed, censored = [], []
    frames = []

    for idx, (model, disk_id) in enumerate(drives):
        vendor = _vendor_of(model)
        live_cols = set(vendor_cols(vendor))

        # ---- lifespan -----------------------------------------
        if idx in fail_idx:
            # fail late enough that a full horizon fits before it
            lo = CFG.horizon_days + CFG.window_len + 5
            fail_day = int(rng.integers(lo, n_days))
            last_day = fail_day
            failure_date = START + timedelta(days=fail_day)
            failed.append((model, disk_id, failure_date))
        elif idx in cens_idx:
            last_day = int(rng.integers(n_days // 3, n_days - 10))
            failure_date = None
            censored.append((model, disk_id,
                             START + timedelta(days=last_day)))
        else:
            last_day = n_days - 1
            failure_date = None

        n_obs = last_day + 1
        dates = [START + timedelta(days=d) for d in range(n_obs)]

        # ---- SMART values -------------------------------------
        cols: dict[str, np.ndarray] = {}
        for c in SMART_COLS:
            if c not in live_cols:
                cols[c] = np.full(n_obs, np.nan)
                continue

            if c.startswith("n_"):
                base = rng.uniform(90, 100)
                series = base - np.linspace(0, rng.uniform(0, 4), n_obs)
                series += rng.normal(0, 0.4, n_obs)
            else:
                base = rng.uniform(0, 500)
                drift = np.cumsum(rng.exponential(0.5, n_obs))
                series = base + drift + rng.normal(0, 2, n_obs)

            cols[c] = series

        # ---- inject pre-failure signal ------------------------
        if failure_date is not None:
            h = CFG.horizon_days
            k = min(h, n_obs)
            ramp = np.zeros(n_obs)
            ramp[-k:] = np.linspace(0, 1, k) ** 2

            for sid in SIGNAL_IDS:
                nc, rc = f"n_{sid}", f"r_{sid}"
                if nc in live_cols:
                    cols[nc] = cols[nc] - 25 * ramp
                if rc in live_cols:
                    cols[rc] = cols[rc] + 400 * ramp

        # ---- sporadic missingness -----------------------------
        for c in live_cols:
            gaps = rng.random(n_obs) < 0.01
            cols[c] = np.where(gaps, np.nan, cols[c])

        frame = pl.DataFrame({
            "disk_id": np.full(n_obs, disk_id, dtype=np.int64),
            "ds": dates,
            "model": [model] * n_obs,
            **{c: cols[c].astype(np.float64) for c in SMART_COLS},
        })
        frames.append(frame)

    df = pl.concat(frames, how="vertical").select(ALL_COLS)

    # -- join failure labels, mirroring preprocess_data.py -------
    if failed:
        fail_df = pl.DataFrame(
            {
                "model": [m for m, _, _ in failed],
                "disk_id": [d for _, d, _ in failed],
                "failure_time": [t for _, _, t in failed],
            },
            schema={"model": pl.String,
                    "disk_id": pl.Int64,
                    "failure_time": pl.Date},
        )
    else:
        fail_df = pl.DataFrame(
            schema={"model": pl.String,
                    "disk_id": pl.Int64,
                    "failure_time": pl.Date}
        )

    df = (
        df.join(fail_df, on=["model", "disk_id"], how="left")
        .with_columns(
            (pl.col("failure_time") - pl.col("ds"))
            .dt.total_days().cast(pl.Float64).alias("days_to_failure")
        )
        .with_columns(
            (pl.col("days_to_failure") <= 0)
            .fill_null(False).cast(pl.Int8).alias("failure_event")
        )
    )

    truth = {
        "failed": failed,
        "censored": censored,
        "signal_ids": SIGNAL_IDS,
        "vendor_smart_ids": VENDOR_SMART_IDS,
        "global_end": global_end,
        "n_drives": len(drives),
        "n_failed": len(failed),
        "horizon_days": CFG.horizon_days,
        "seed": seed,
    }
    return df, truth


def write_synthetic(path: str, **kwargs) -> dict:
    """Write a synthetic fixture to parquet. Returns ground truth."""
    df, truth = make_synthetic(**kwargs)
    df.write_parquet(path)
    return truth
