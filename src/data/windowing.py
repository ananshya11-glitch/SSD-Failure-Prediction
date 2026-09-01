"""
Trajectory windowing.

Turns labelled daily SMART rows into fixed-length sequences, the actual
model input.

    (n_samples, window_len, n_features)

WINDOW SEMANTICS
    A window ending at day T is "everything known as of T". Its label is
    the label of row T: does this drive fail within `horizon` days of T.
    The label is therefore taken from the LAST row of the window, never
    the first and never an aggregate. Taking it from anywhere else
    silently shifts the prediction horizon.

STRIDE
    `stride` subsamples windows along time. `positive_stride` defaults
    to `stride`, i.e. NO oversampling of the positive class.

    Oversampling positives is available but off by default, and should
    stay off for the headline results. The paper's central measurement
    is that failure-class coverage collapses under low prevalence;
    inflating prevalence at the windowing stage would mask exactly the
    effect being reported. Setting positive_stride=1 with stride=7 on
    the synthetic fixture lifts prevalence from 4% to 23% -- an
    artefact, not a finding.

    Coverage guarantees are unaffected either way, since the same
    windowing is applied to calibration and test, so exchangeability
    within a split is preserved. What changes is the operational
    meaning of the numbers. If oversampling is used for training, the
    conformal layer must still calibrate on naturally-sampled windows.

DATE GAPS
    Drives skip days. Windows are built over a REINDEXED continuous
    daily range per drive, so a window always spans exactly
    `window_len` calendar days. Windowing over positional index instead
    would let a window silently span months.

MISSING VALUES
    Reindexing introduces gaps, and the raw data already has NaN. Values
    are forward-filled within a drive, then zero-filled at the head. A
    window is rejected if the fraction of originally-present values
    falls below `min_valid_frac`, so a window that is mostly imputation
    never reaches the model.

NORMALISATION
    Statistics are fit on TRAINING drives only and applied unchanged to
    calibration and test. Fitting on pooled data leaks test information
    into the model and, worse, into the conformal calibration scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import CFG, DRIVE_KEY
from ..schema import COMMON_COLS
from .labels import LABEL_COL, VENDOR_COL


@dataclass
class WindowSet:
    """
    Windowed data plus the provenance needed to group by drive.

    X          (n, window_len, n_features) float32
    y          (n,) int8
    drive_idx  (n,) int  -- index into `drives`
    end_ds     (n,) date -- last day of each window
    drives     list[(model, disk_id)]
    vendors    (n,) str  -- vendor of each window's drive
    features   list[str]
    """
    X: np.ndarray
    y: np.ndarray
    drive_idx: np.ndarray
    end_ds: np.ndarray
    drives: list[tuple]
    vendors: np.ndarray
    features: list[str]
    window_len: int
    stride: int
    positive_stride: int

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    @property
    def n_drives(self) -> int:
        return len(self.drives)

    def drive_keys(self) -> np.ndarray:
        """(n,) array of (model, disk_id) tuples, one per window."""
        return np.array([self.drives[i] for i in self.drive_idx],
                        dtype=object)

    def summary(self) -> dict:
        return {
            "n_windows": len(self),
            "n_drives": self.n_drives,
            "n_features": len(self.features),
            "window_len": self.window_len,
            "n_positive": int(self.y.sum()),
            "positive_rate": self.positive_rate,
            "stride": self.stride,
            "positive_stride": self.positive_stride,
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"{s['n_windows']:,} windows from {s['n_drives']:,} drives\n"
            f"  shape      : {self.X.shape}\n"
            f"  positives  : {s['n_positive']:,} "
            f"({100 * s['positive_rate']:.4f}%)\n"
            f"  stride     : {s['stride']} "
            f"(positives {s['positive_stride']})"
        )


def _reindex_drive(g: pl.DataFrame, features: list[str]) -> tuple:
    """
    Reindex one drive to a continuous daily range.

    Returns (values, labels, dates, present_mask) where present_mask
    marks positions that came from a real observation with a non-null
    value, before any filling.
    """
    first, last = g["ds"].min(), g["ds"].max()
    full = pl.DataFrame({
        "ds": pl.date_range(first, last, interval="1d", eager=True)
    })
    j = full.join(g, on="ds", how="left").sort("ds")

    vals = j.select(features).to_numpy().astype(np.float64)
    present = np.isfinite(vals)

    # forward fill within the drive, then zero the leading gap
    for c in range(vals.shape[1]):
        col = vals[:, c]
        idx = np.where(np.isfinite(col), np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        filled[~np.isfinite(filled)] = 0.0
        vals[:, c] = filled

    labels = j[LABEL_COL].fill_null(-1).to_numpy().astype(np.int64)
    # to_list() keeps datetime.date; to_numpy() would give datetime64,
    # which does not compare or subtract cleanly against the dates
    # carried everywhere else in the pipeline.
    dates = j["ds"].to_list()
    return vals, labels, dates, present


def make_windows(
    labelled: pl.DataFrame,
    features: list[str] | None = None,
    window_len: int | None = None,
    stride: int | None = None,
    positive_stride: int | None = None,
    min_valid_frac: float | None = None,
) -> WindowSet:
    """
    Build fixed-length windows from labelled daily rows.

    `features` defaults to the common-16 set, per the locked feature
    policy: only those columns are populated by every vendor, so only
    they are usable in every LOMO fold.

    Rows whose label is undefined (absent after reindexing) can appear
    inside a window as inputs, but a window is never emitted for an
    undefined end row.
    """
    features = list(features) if features is not None else list(COMMON_COLS)
    window_len = window_len if window_len is not None else CFG.window_len
    stride = stride if stride is not None else CFG.stride
    min_valid_frac = (min_valid_frac if min_valid_frac is not None
                      else CFG.min_valid_frac)
    # Default: no positive oversampling. See the STRIDE note above.
    positive_stride = (positive_stride if positive_stride is not None
                       else stride)

    missing = [c for c in features if c not in labelled.columns]
    if missing:
        raise ValueError(f"features not in frame: {missing[:5]}")

    Xs, ys, didx, ends, vends = [], [], [], [], []
    drives: list[tuple] = []

    for key, g in labelled.sort(DRIVE_KEY + ["ds"]).group_by(
        DRIVE_KEY, maintain_order=True
    ):
        g = g.sort("ds")
        vendor = g[VENDOR_COL][0]
        vals, labels, dates, present = _reindex_drive(g, features)
        n = vals.shape[0]
        if n < window_len:
            continue

        di = len(drives)
        drives.append(tuple(key))
        emitted = False

        # candidate end positions, newest-independent
        for end in range(window_len - 1, n):
            lab = labels[end]
            if lab < 0:
                continue  # end row has no observed label

            step = positive_stride if lab == 1 else stride
            # phase the stride off the drive's first valid end position
            if (end - (window_len - 1)) % step != 0:
                continue

            s = end - window_len + 1
            win_present = present[s:end + 1]
            if win_present.mean() < min_valid_frac:
                continue

            Xs.append(vals[s:end + 1])
            ys.append(lab)
            didx.append(di)
            ends.append(dates[end])
            vends.append(vendor)
            emitted = True

        if not emitted:
            drives.pop()

    if not Xs:
        return WindowSet(
            X=np.zeros((0, window_len, len(features)), dtype=np.float32),
            y=np.zeros(0, dtype=np.int8),
            drive_idx=np.zeros(0, dtype=int),
            end_ds=np.zeros(0, dtype=object),
            drives=[],
            vendors=np.zeros(0, dtype=object),
            features=features,
            window_len=window_len,
            stride=stride,
            positive_stride=positive_stride,
        )

    return WindowSet(
        X=np.stack(Xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int8),
        drive_idx=np.asarray(didx, dtype=int),
        end_ds=np.asarray(ends, dtype=object),
        drives=drives,
        vendors=np.asarray(vends, dtype=object),
        features=features,
        window_len=window_len,
        stride=stride,
        positive_stride=positive_stride,
    )


# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------

@dataclass
class Normalizer:
    """
    Per-feature standardisation, fit on training windows only.

    Fitting on pooled data leaks test information into both the model
    and the conformal calibration scores, which would inflate coverage
    for the wrong reason.
    """
    mean: np.ndarray
    std: np.ndarray
    features: list[str]

    @classmethod
    def fit(cls, ws: WindowSet, eps: float = 1e-8) -> "Normalizer":
        flat = ws.X.reshape(-1, ws.X.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < eps] = 1.0
        return cls(mean=mean, std=std, features=list(ws.features))

    def transform(self, ws: WindowSet) -> WindowSet:
        if list(ws.features) != list(self.features):
            raise ValueError(
                "feature mismatch between normalizer and window set"
            )
        out = (ws.X - self.mean) / self.std
        return WindowSet(
            X=out.astype(np.float32), y=ws.y, drive_idx=ws.drive_idx,
            end_ds=ws.end_ds, drives=ws.drives, vendors=ws.vendors,
            features=ws.features, window_len=ws.window_len,
            stride=ws.stride, positive_stride=ws.positive_stride,
        )


# ---------------------------------------------------------------
# Flattening, for the Random Forest baseline
# ---------------------------------------------------------------

def flatten(ws: WindowSet, how: str = "last") -> tuple[np.ndarray, list[str]]:
    """
    Collapse the time axis for models that take tabular input.

    "last"  -- the final timestep only, closest to WEFR's per-day rows
    "stats" -- mean, std, min, max, last and the first-to-last delta
    "full"  -- concatenate every timestep
    """
    n, w, f = ws.X.shape
    if how == "last":
        return ws.X[:, -1, :], list(ws.features)
    if how == "full":
        names = [f"{c}_t{t}" for t in range(w) for c in ws.features]
        return ws.X.reshape(n, w * f), names
    if how == "stats":
        parts = [ws.X.mean(1), ws.X.std(1), ws.X.min(1), ws.X.max(1),
                 ws.X[:, -1, :], ws.X[:, -1, :] - ws.X[:, 0, :]]
        names = [f"{c}_{s}" for s in
                 ("mean", "std", "min", "max", "last", "delta")
                 for c in ws.features]
        return np.concatenate(parts, axis=1), names
    raise ValueError(f"unknown how={how!r}")
