"""
30_drive_aggregation_sensitivity.py

Sensitivity analysis for drive-level aggregation.

Compares three ways of converting window-level CNN probabilities
into ONE score per SSD:

    1. last  = probability of the latest window
    2. mean  = mean probability of the final 30 windows
    3. max   = maximum probability of the final 30 windows

Evaluates:
    - STANDARD
    - LOMO-A
    - LOMO-B
    - LOMO-C

No model training is performed.
Existing prediction files are only read.

For conformal prediction:
    - calibration and test use the SAME aggregation rule
    - LAC qhat is calibrated separately for each fold/rule
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

STANDARD_WINDOWS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "standard"
    / "windows"
)

LOMO_WINDOWS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "lomo"
    / "windows"
)

REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.10

FINAL_WINDOWS = 30

AGGREGATIONS = (
    "last",
    "mean",
    "max",
)


# ============================================================
# LOAD STANDARD
# ============================================================

def load_standard(name: str):

    prob_path = (
        RESULTS_DIR
        / f"standard_cnn_{name}_probs.npy"
    )

    label_path = (
        RESULTS_DIR
        / f"standard_cnn_{name}_labels.npy"
    )

    metadata_path = (
        STANDARD_WINDOWS_DIR
        / f"{name}_metadata.parquet"
    )

    print()
    print("=" * 70)
    print(f"STANDARD {name.upper()}")
    print("=" * 70)

    print(f"Probability : {prob_path}")
    print(f"Labels      : {label_path}")
    print(f"Metadata    : {metadata_path}")

    if not prob_path.exists():
        raise FileNotFoundError(prob_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    probs = np.load(
        prob_path
    ).astype(np.float64)

    labels = np.load(
        label_path
    ).astype(np.int8)

    metadata = pl.read_parquet(
        metadata_path
    )

    required = {
        "model",
        "disk_id",
        "window_date",
        "label",
    }

    missing = (
        required
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(probs) != len(labels):
        raise ValueError(
            "Probability and label lengths differ."
        )

    if metadata.height != len(probs):
        raise ValueError(
            "Metadata and prediction lengths differ."
        )

    metadata_labels = (
        metadata["label"]
        .to_numpy()
        .astype(np.int8)
    )

    if not np.array_equal(
        metadata_labels,
        labels,
    ):
        raise AssertionError(
            "Metadata labels do not match "
            "saved labels."
        )

    return metadata, probs, labels


# ============================================================
# LOAD LOMO
# ============================================================

def load_lomo(
    fold: str,
    name: str,
):

    prob_path = (
        RESULTS_DIR
        / "lomo"
        / f"lomo_{fold}_{name}_probs.npy"
    )

    label_path = (
        RESULTS_DIR
        / "lomo"
        / f"lomo_{fold}_{name}_labels.npy"
    )

    metadata_path = (
        LOMO_WINDOWS_DIR
        / f"lomo_{fold}"
        / name
        / "metadata.parquet"
    )

    print()
    print("=" * 70)
    print(f"LOMO-{fold} {name.upper()}")
    print("=" * 70)

    print(f"Probability : {prob_path}")
    print(f"Labels      : {label_path}")
    print(f"Metadata    : {metadata_path}")

    if not prob_path.exists():
        raise FileNotFoundError(prob_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    probs = np.load(
        prob_path
    ).astype(np.float64)

    labels = np.load(
        label_path
    ).astype(np.int8)

    metadata = pl.read_parquet(
        metadata_path
    )

    required = {
        "model",
        "disk_id",
        "window_date",
        "label",
    }

    missing = (
        required
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(probs) != len(labels):
        raise ValueError(
            "Probability and label lengths differ."
        )

    if metadata.height != len(probs):
        raise ValueError(
            "Metadata and prediction lengths differ."
        )

    metadata_labels = (
        metadata["label"]
        .to_numpy()
        .astype(np.int8)
    )

    if not np.array_equal(
        metadata_labels,
        labels,
    ):
        raise AssertionError(
            "Metadata labels do not match "
            "saved labels."
        )

    return metadata, probs, labels


# ============================================================
# DRIVE AGGREGATION
# ============================================================

def make_drive_scores(
    metadata: pl.DataFrame,
    probs: np.ndarray,
    aggregation: str,
):
    """
    Produce exactly one score per drive.

    The final 30 windows are selected first.

    aggregation:

        last:
            latest probability

        mean:
            mean probability of final 30 windows

        max:
            maximum probability of final 30 windows
    """

    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    frame = metadata.with_columns(
        pl.Series(
            name="probability",
            values=probs,
        )
    )

    frame = frame.sort(
        [
            "model",
            "disk_id",
            "window_date",
        ]
    )

    # --------------------------------------------------------
    # Final 30 windows
    # --------------------------------------------------------

    final_windows = (
        frame
        .group_by(
            [
                "model",
                "disk_id",
            ],
            maintain_order=True,
        )
        .tail(FINAL_WINDOWS)
    )

    # --------------------------------------------------------
    # Aggregation expression
    # --------------------------------------------------------

    if aggregation == "last":

        score_expr = (
            pl.col("probability")
            .last()
            .alias("drive_score")
        )

    elif aggregation == "mean":

        score_expr = (
            pl.col("probability")
            .mean()
            .alias("drive_score")
        )

    elif aggregation == "max":

        score_expr = (
            pl.col("probability")
            .max()
            .alias("drive_score")
        )

    # --------------------------------------------------------
    # One row per drive
    # --------------------------------------------------------

    drives = (
        final_windows
        .group_by(
            [
                "model",
                "disk_id",
            ],
            maintain_order=True,
        )
        .agg(
            [
                score_expr,

                pl.col("label")
                .max()
                .cast(pl.Int8)
                .alias("drive_label"),

                pl.len()
                .alias("n_final_windows"),

                pl.col("window_date")
                .min()
                .alias("final_window_start"),

                pl.col("window_date")
                .max()
                .alias("final_window_end"),
            ]
        )
        .sort(
            [
                "model",
                "disk_id",
            ]
        )
    )

    if drives.height == 0:
        raise ValueError(
            "No drive-level rows produced."
        )

    # --------------------------------------------------------
    # Exactly one row per drive
    # --------------------------------------------------------

    duplicate_count = (
        drives
        .group_by(
            [
                "model",
                "disk_id",
            ]
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    if duplicate_count != 0:
        raise AssertionError(
            "Duplicate drive-level rows detected."
        )

    return drives


# ============================================================
# CLASSIFICATION
# ============================================================

def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
):

    if len(np.unique(labels)) == 2:

        roc_auc = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        pr_auc = float(
            average_precision_score(
                labels,
                scores,
            )
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")

    predictions = (
        scores >= 0.5
    ).astype(np.int8)

    precision = float(
        precision_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# LAC QHAT
# ============================================================

def get_lac_qhat(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
):

    nonconformity = np.where(
        labels == 1,
        1.0 - scores,
        scores,
    )

    n = len(
        nonconformity
    )

    if n == 0:
        raise ValueError(
            "Empty calibration set."
        )

    rank = int(
        np.ceil(
            (n + 1)
            * (1.0 - alpha)
        )
    )

    rank = max(
        1,
        min(rank, n),
    )

    qhat = float(
        np.sort(
            nonconformity
        )[rank - 1]
    )

    return qhat


# ============================================================
# LAC METRICS
# ============================================================

def get_lac_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    qhat: float,
):

    p1 = scores
    p0 = 1.0 - scores

    include_0 = (
        p0 <= qhat
    )

    include_1 = (
        p1 <= qhat
    )

    covered = np.where(
        labels == 1,
        include_1,
        include_0,
    )

    set_size = (
        include_0.astype(np.int8)
        +
        include_1.astype(np.int8)
    )

    healthy = (
        labels == 0
    )

    failure = (
        labels == 1
    )

    return {
        "coverage":
            float(np.mean(covered)),

        "coverage_healthy":
            float(
                np.mean(
                    covered[healthy]
                )
            ),

        "coverage_failure":
            float(
                np.mean(
                    covered[failure]
                )
            ),

        "avg_set_size":
            float(
                np.mean(set_size)
            ),

        "singleton_rate":
            float(
                np.mean(
                    set_size == 1
                )
            ),

        "doubleton_rate":
            float(
                np.mean(
                    set_size == 2
                )
            ),

        "empty_rate":
            float(
                np.mean(
                    set_size == 0
                )
            ),
    }


# ============================================================
# PROCESS ONE EXPERIMENT
# ============================================================

def process_experiment(
    experiment: str,
    fold: str | None,
    cal_metadata: pl.DataFrame,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_metadata: pl.DataFrame,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
):

    rows = []

    for aggregation in AGGREGATIONS:

        # ----------------------------------------------------
        # Drive-level calibration
        # ----------------------------------------------------

        cal_drives = make_drive_scores(
            cal_metadata,
            cal_probs,
            aggregation,
        )

        cal_scores = (
            cal_drives["drive_score"]
            .to_numpy()
            .astype(np.float64)
        )

        cal_y = (
            cal_drives["drive_label"]
            .to_numpy()
            .astype(np.int8)
        )

        # ----------------------------------------------------
        # Drive-level test
        # ----------------------------------------------------

        test_drives = make_drive_scores(
            test_metadata,
            test_probs,
            aggregation,
        )

        test_scores = (
            test_drives["drive_score"]
            .to_numpy()
            .astype(np.float64)
        )

        test_y = (
            test_drives["drive_label"]
            .to_numpy()
            .astype(np.int8)
        )

        if len(np.unique(cal_y)) != 2:
            raise ValueError(
                f"{experiment} {aggregation}: "
                "calibration lacks both classes."
            )

        if len(np.unique(test_y)) != 2:
            raise ValueError(
                f"{experiment} {aggregation}: "
                "test lacks both classes."
            )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        classification = (
            classification_metrics(
                test_y,
                test_scores,
            )
        )

        # ----------------------------------------------------
        # LAC
        # ----------------------------------------------------

        qhat = get_lac_qhat(
            cal_scores,
            cal_y,
            ALPHA,
        )

        conformal = get_lac_metrics(
            test_scores,
            test_y,
            qhat,
        )

        # ----------------------------------------------------
        # Save drive scores
        # ----------------------------------------------------

        if fold is None:

            filename = (
                f"standard_drive_scores_"
                f"{aggregation}.csv"
            )

        else:

            filename = (
                f"lomo_{fold}_drive_scores_"
                f"{aggregation}.csv"
            )

        test_drives.write_csv(
            REPORTS_DIR / filename
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        row = {
            "experiment":
                experiment,

            "fold":
                fold if fold is not None
                else "standard",

            "aggregation":
                aggregation,

            "n_cal_drives":
                len(cal_y),

            "n_cal_failed_drives":
                int(cal_y.sum()),

            "n_test_drives":
                len(test_y),

            "n_test_failed_drives":
                int(test_y.sum()),

            "test_prevalence":
                float(
                    np.mean(test_y)
                ),

            "roc_auc":
                classification["roc_auc"],

            "pr_auc":
                classification["pr_auc"],

            "precision":
                classification["precision"],

            "recall":
                classification["recall"],

            "qhat_lac":
                qhat,

            "coverage_lac":
                conformal["coverage"],

            "coverage_healthy_lac":
                conformal[
                    "coverage_healthy"
                ],

            "coverage_failure_lac":
                conformal[
                    "coverage_failure"
                ],

            "avg_set_size_lac":
                conformal[
                    "avg_set_size"
                ],

            "singleton_rate_lac":
                conformal[
                    "singleton_rate"
                ],

            "doubleton_rate_lac":
                conformal[
                    "doubleton_rate"
                ],

            "empty_rate_lac":
                conformal[
                    "empty_rate"
                ],
        }

        rows.append(row)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DRIVE-LEVEL AGGREGATION SENSITIVITY ANALYSIS")
    print("=" * 70)

    all_rows = []

    # ========================================================
    # STANDARD
    # ========================================================

    (
        standard_cal_metadata,
        standard_cal_probs,
        standard_cal_labels,
    ) = load_standard(
        "calibration"
    )

    (
        standard_test_metadata,
        standard_test_probs,
        standard_test_labels,
    ) = load_standard(
        "test"
    )

    standard_rows = process_experiment(
        experiment="standard",
        fold=None,

        cal_metadata=standard_cal_metadata,
        cal_probs=standard_cal_probs,
        cal_labels=standard_cal_labels,

        test_metadata=standard_test_metadata,
        test_probs=standard_test_probs,
        test_labels=standard_test_labels,
    )

    all_rows.extend(
        standard_rows
    )

    # ========================================================
    # LOMO
    # ========================================================

    for fold in (
        "A",
        "B",
        "C",
    ):

        (
            cal_metadata,
            cal_probs,
            cal_labels,
        ) = load_lomo(
            fold,
            "calibration",
        )

        (
            test_metadata,
            test_probs,
            test_labels,
        ) = load_lomo(
            fold,
            "test",
        )

        rows = process_experiment(
            experiment=f"lomo_{fold}",
            fold=fold,

            cal_metadata=cal_metadata,
            cal_probs=cal_probs,
            cal_labels=cal_labels,

            test_metadata=test_metadata,
            test_probs=test_probs,
            test_labels=test_labels,
        )

        all_rows.extend(
            rows
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results = pl.DataFrame(
        all_rows
    )

    result_path = (
        REPORTS_DIR
        / "drive_aggregation_sensitivity.csv"
    )

    results.write_csv(
        result_path
    )

    # ========================================================
    # PRINT CLASSIFICATION SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION SENSITIVITY")
    print("=" * 70)

    print(
        results.select(
            [
                "experiment",
                "aggregation",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
            ]
        )
    )

    # ========================================================
    # PRINT CONFORMAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CONFORMAL SENSITIVITY")
    print("=" * 70)

    print(
        results.select(
            [
                "experiment",
                "aggregation",
                "qhat_lac",
                "coverage_lac",
                "coverage_healthy_lac",
                "coverage_failure_lac",
                "avg_set_size_lac",
                "empty_rate_lac",
            ]
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Combined results:\n{result_path}"
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    main()