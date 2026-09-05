"""
29_make_standard_drive_level_conformal.py

Drive-level evaluation for the STANDARD CNN experiment.

Method:
    1. Load the existing window-level CNN probabilities.
    2. Load the already-generated STANDARD window metadata.
    3. For each SSD, take its final 30 prediction windows.
    4. Average their probabilities to obtain ONE score per SSD.
    5. Use the same aggregation for calibration and test.
    6. Perform drive-level LAC conformal evaluation.

IMPORTANT:
    - No model training.
    - No modification of existing prediction files.
    - Uses the existing STANDARD metadata.
    - Each SSD contributes exactly one score.
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

WINDOWS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "standard"
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

# Primary drive-level aggregation:
#
# Take the 30 latest prediction windows available for each
# drive and calculate their mean probability.
#
# If a drive has fewer than 30 windows, all available windows
# are used.
FINAL_WINDOWS = 30


# ============================================================
# LOAD STANDARD DATA
# ============================================================

def load_set(name: str):
    """
    Load one STANDARD split.

    Valid names:
        calibration
        test

    The actual verified STANDARD metadata filenames are:

        calibration_metadata.parquet
        test_metadata.parquet
        train_metadata.parquet
    """

    prob_path = (
        RESULTS_DIR
        / f"standard_cnn_{name}_probs.npy"
    )

    label_path = (
        RESULTS_DIR
        / f"standard_cnn_{name}_labels.npy"
    )

    # IMPORTANT:
    # STANDARD metadata files are directly inside /windows/.
    metadata_path = (
        WINDOWS_DIR
        / f"{name}_metadata.parquet"
    )

    print()
    print("=" * 70)
    print(f"STANDARD {name.upper()}")
    print("=" * 70)

    print(
        f"Probability file : {prob_path}"
    )

    print(
        f"Label file       : {label_path}"
    )

    print(
        f"Metadata file    : {metadata_path}"
    )

    # --------------------------------------------------------
    # File existence checks
    # --------------------------------------------------------

    if not prob_path.exists():
        raise FileNotFoundError(
            f"Probability file not found:\n{prob_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found:\n{label_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_path}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    probs = np.load(
        prob_path
    ).astype(np.float64)

    labels = np.load(
        label_path
    ).astype(np.int8)

    metadata = pl.read_parquet(
        metadata_path
    )

    # --------------------------------------------------------
    # Verify schema
    # --------------------------------------------------------

    required_columns = {
        "model",
        "disk_id",
        "window_date",
        "label",
    }

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            "STANDARD metadata is missing "
            f"columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Verify lengths
    # --------------------------------------------------------

    if len(probs) != len(labels):
        raise ValueError(
            "Probability/label length mismatch: "
            f"{len(probs)} vs {len(labels)}"
        )

    if metadata.height != len(probs):
        raise ValueError(
            "Metadata/prediction length mismatch: "
            f"{metadata.height} vs {len(probs)}"
        )

    # --------------------------------------------------------
    # Verify metadata labels match saved labels
    # --------------------------------------------------------

    metadata_labels = (
        metadata["label"]
        .to_numpy()
        .astype(np.int8)
    )

    if not np.array_equal(
        metadata_labels,
        labels,
    ):
        mismatches = int(
            np.sum(
                metadata_labels != labels
            )
        )

        raise AssertionError(
            "Metadata labels do not match "
            f"saved labels. Mismatches: {mismatches}"
        )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"Windows          : {len(probs):,}"
    )

    print(
        f"Positive windows : {int(labels.sum()):,}"
    )

    print(
        f"Metadata rows    : {metadata.height:,}"
    )

    print(
        f"Metadata columns : {metadata.columns}"
    )

    return (
        metadata,
        probs,
        labels,
    )


# ============================================================
# DRIVE-LEVEL AGGREGATION
# ============================================================

def make_drive_scores(
    metadata: pl.DataFrame,
    probs: np.ndarray,
):
    """
    Convert window-level predictions into exactly one
    prediction score per SSD.

    For each drive:

        1. Sort windows chronologically.
        2. Keep the final 30 windows.
        3. Average their probabilities.

    Drive label:

        1 if ANY of the drive's windows has label 1.

    This produces exactly one row per drive.
    """

    # --------------------------------------------------------
    # Add probability to metadata
    # --------------------------------------------------------

    frame = metadata.with_columns(
        pl.Series(
            name="probability",
            values=probs,
        )
    )

    # --------------------------------------------------------
    # Chronological order within each drive
    # --------------------------------------------------------

    frame = frame.sort(
        [
            "model",
            "disk_id",
            "window_date",
        ]
    )

    # --------------------------------------------------------
    # Take final 30 windows for each drive
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
    # Aggregate to ONE row per drive
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
                pl.col("probability")
                .mean()
                .alias("drive_score"),

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

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if drives.height == 0:
        raise ValueError(
            "Drive-level aggregation produced zero drives."
        )

    # Each drive must occur exactly once.
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
            "Drive-level aggregation produced "
            f"{duplicate_count} duplicate drives."
        )

    # --------------------------------------------------------
    # Validate drive labels
    # --------------------------------------------------------

    drive_labels = (
        drives["drive_label"]
        .to_numpy()
        .astype(np.int8)
    )

    if not np.all(
        np.isin(
            drive_labels,
            [0, 1],
        )
    ):
        raise AssertionError(
            "Drive labels contain values other than 0/1."
        )

    return drives


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
):
    """
    Calculate drive-level ranking and threshold metrics.

    Threshold:
        probability >= 0.5 -> failure

    ROC-AUC and PR-AUC are threshold-independent.
    """

    unique_labels = np.unique(
        labels
    )

    if len(unique_labels) == 2:

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
# LAC CONFORMAL CALIBRATION
# ============================================================

def get_lac_qhat(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    alpha: float,
):
    """
    Binary Least Ambiguous set-valued conformal prediction.

    Nonconformity scores:

        true class = 1:
            score = 1 - p

        true class = 0:
            score = p

    Finite-sample quantile:

        ceil((n + 1) * (1 - alpha))
    """

    calibration_scores = np.asarray(
        calibration_scores,
        dtype=np.float64,
    )

    calibration_labels = np.asarray(
        calibration_labels,
        dtype=np.int8,
    )

    if len(calibration_scores) == 0:
        raise ValueError(
            "Calibration set is empty."
        )

    # --------------------------------------------------------
    # Nonconformity scores
    # --------------------------------------------------------

    scores = np.where(
        calibration_labels == 1,
        1.0 - calibration_scores,
        calibration_scores,
    )

    # --------------------------------------------------------
    # Finite-sample rank
    # --------------------------------------------------------

    n = len(scores)

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

    # --------------------------------------------------------
    # Quantile
    # --------------------------------------------------------

    sorted_scores = np.sort(
        scores
    )

    qhat = float(
        sorted_scores[rank - 1]
    )

    return qhat


# ============================================================
# LAC PREDICTION SETS
# ============================================================

def make_lac_sets(
    scores: np.ndarray,
    qhat: float,
):
    """
    Construct binary LAC prediction sets.

    Class 0 is included when:

        1 - p <= qhat

    Class 1 is included when:

        p <= qhat
    """

    p1 = scores
    p0 = 1.0 - scores

    include_0 = (
        p0 <= qhat
    )

    include_1 = (
        p1 <= qhat
    )

    return (
        include_0,
        include_1,
    )


# ============================================================
# CONFORMAL METRICS
# ============================================================

def conformal_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    qhat: float,
):
    """
    Calculate:

        overall coverage
        healthy coverage
        failure coverage
        average prediction-set size
        singleton rate
        doubleton rate
        empty-set rate
    """

    include_0, include_1 = (
        make_lac_sets(
            scores,
            qhat,
        )
    )

    # --------------------------------------------------------
    # Was the true class included?
    # --------------------------------------------------------

    covered = np.where(
        labels == 1,
        include_1,
        include_0,
    )

    # --------------------------------------------------------
    # Prediction-set size
    # --------------------------------------------------------

    set_size = (
        include_0.astype(np.int8)
        +
        include_1.astype(np.int8)
    )

    healthy_mask = (
        labels == 0
    )

    failure_mask = (
        labels == 1
    )

    # --------------------------------------------------------
    # Overall coverage
    # --------------------------------------------------------

    coverage = float(
        np.mean(covered)
    )

    # --------------------------------------------------------
    # Healthy coverage
    # --------------------------------------------------------

    if np.any(healthy_mask):

        coverage_healthy = float(
            np.mean(
                covered[
                    healthy_mask
                ]
            )
        )

    else:

        coverage_healthy = float(
            "nan"
        )

    # --------------------------------------------------------
    # Failure coverage
    # --------------------------------------------------------

    if np.any(failure_mask):

        coverage_failure = float(
            np.mean(
                covered[
                    failure_mask
                ]
            )
        )

    else:

        coverage_failure = float(
            "nan"
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "coverage": coverage,

        "coverage_healthy":
            coverage_healthy,

        "coverage_failure":
            coverage_failure,

        "avg_set_size": float(
            np.mean(set_size)
        ),

        "singleton_rate": float(
            np.mean(
                set_size == 1
            )
        ),

        "doubleton_rate": float(
            np.mean(
                set_size == 2
            )
        ),

        "empty_rate": float(
            np.mean(
                set_size == 0
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STANDARD DRIVE-LEVEL CONFORMAL EVALUATION")
    print("=" * 70)

    # ========================================================
    # CALIBRATION
    # ========================================================

    (
        cal_metadata,
        cal_probs,
        cal_labels,
    ) = load_set(
        "calibration"
    )

    cal_drives = make_drive_scores(
        cal_metadata,
        cal_probs,
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

    # ========================================================
    # TEST
    # ========================================================

    (
        test_metadata,
        test_probs,
        test_labels,
    ) = load_set(
        "test"
    )

    test_drives = make_drive_scores(
        test_metadata,
        test_probs,
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

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(np.unique(cal_y)) != 2:
        raise ValueError(
            "Calibration drive set does not contain "
            "both healthy and failed drives."
        )

    if len(np.unique(test_y)) != 2:
        raise ValueError(
            "Test drive set does not contain "
            "both healthy and failed drives."
        )

    # ========================================================
    # DRIVE COUNTS
    # ========================================================

    print()
    print("DRIVE COUNTS")
    print("-" * 70)

    print(
        f"Calibration drives : "
        f"{len(cal_y):,}"
    )

    print(
        f"Calibration failed : "
        f"{int(cal_y.sum()):,}"
    )

    print(
        f"Calibration healthy: "
        f"{int((cal_y == 0).sum()):,}"
    )

    print(
        f"Test drives        : "
        f"{len(test_y):,}"
    )

    print(
        f"Test failed        : "
        f"{int(test_y.sum()):,}"
    )

    print(
        f"Test healthy       : "
        f"{int((test_y == 0).sum()):,}"
    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    metrics = classification_metrics(
        test_y,
        test_scores,
    )

    # ========================================================
    # LAC CALIBRATION
    # ========================================================

    qhat = get_lac_qhat(
        cal_scores,
        cal_y,
        ALPHA,
    )

    # ========================================================
    # LAC TEST
    # ========================================================

    conformal = conformal_metrics(
        test_scores,
        test_y,
        qhat,
    )

    # ========================================================
    # SAVE DRIVE-LEVEL SCORES
    # ========================================================

    drive_path = (
        REPORTS_DIR
        / "standard_drive_level_scores.csv"
    )

    test_drives.write_csv(
        drive_path
    )

    # ========================================================
    # BUILD RESULT TABLE
    # ========================================================

    result = pl.DataFrame(
        [
            {
                "split": "standard",

                "aggregation":
                    "mean_final_30_windows",

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
                    metrics["roc_auc"],

                "pr_auc":
                    metrics["pr_auc"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

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
        ]
    )

    result_path = (
        REPORTS_DIR
        / "standard_drive_level_conformal_results.csv"
    )

    result.write_csv(
        result_path
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("STANDARD DRIVE-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC          : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC           : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Precision        : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall           : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"LAC qhat         : "
        f"{qhat:.6f}"
    )

    print(
        f"LAC coverage     : "
        f"{conformal['coverage']:.4f}"
    )

    print(
        f"Healthy coverage : "
        f"{conformal['coverage_healthy']:.4f}"
    )

    print(
        f"Failure coverage : "
        f"{conformal['coverage_failure']:.4f}"
    )

    print(
        f"Average set size : "
        f"{conformal['avg_set_size']:.4f}"
    )

    print(
        f"Singleton rate   : "
        f"{conformal['singleton_rate']:.4f}"
    )

    print(
        f"Doubleton rate   : "
        f"{conformal['doubleton_rate']:.4f}"
    )

    print(
        f"Empty-set rate   : "
        f"{conformal['empty_rate']:.4f}"
    )

    # ========================================================
    # SAVE LOCATIONS
    # ========================================================

    print()
    print(
        f"Drive scores saved : "
        f"{drive_path}"
    )

    print(
        f"Results saved      : "
        f"{result_path}"
    )

    print()
    print("=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()