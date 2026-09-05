"""
28_make_drive_level_conformal.py

Convert existing window-level CNN predictions into ONE score per SSD
and evaluate conformal prediction at the DRIVE level.

IMPORTANT:
- Does NOT retrain any model.
- Does NOT modify existing prediction files.
- Uses the already verified LOMO prediction files and metadata.
- Each SSD contributes exactly one score.
- Primary drive score = mean probability over the drive's final
  30-day prediction window.
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

RESULTS_DIR = ROOT / "results" / "lomo"
WINDOWS_DIR = ROOT / "data" / "processed" / "lomo" / "windows"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.10


# ============================================================
# VERIFIED LOMO FILES
# ============================================================

FOLDS = ("A", "B", "C")


# ============================================================
# HELPERS
# ============================================================

def load_fold(fold: str):
    """
    Load one LOMO test fold.

    Returns:
        metadata
        probabilities
        labels
    """

    prob_path = RESULTS_DIR / f"lomo_{fold}_test_probs.npy"
    label_path = RESULTS_DIR / f"lomo_{fold}_test_labels.npy"
    metadata_path = (
        WINDOWS_DIR
        / f"lomo_{fold}"
        / "test"
        / "metadata.parquet"
    )

    print()
    print("=" * 70)
    print(f"LOMO-{fold}")
    print("=" * 70)

    print(f"Probability file : {prob_path}")
    print(f"Label file       : {label_path}")
    print(f"Metadata file    : {metadata_path}")

    if not prob_path.exists():
        raise FileNotFoundError(prob_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    probs = np.load(prob_path)
    labels = np.load(label_path)
    metadata = pl.read_parquet(metadata_path)

    required = {
        "model",
        "disk_id",
        "window_date",
        "label",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"LOMO-{fold}: metadata missing columns: {sorted(missing)}"
        )

    if len(probs) != len(labels):
        raise ValueError(
            f"LOMO-{fold}: probability/label length mismatch: "
            f"{len(probs)} vs {len(labels)}"
        )

    if metadata.height != len(probs):
        raise ValueError(
            f"LOMO-{fold}: metadata/prediction length mismatch: "
            f"{metadata.height} vs {len(probs)}"
        )

    # Verify the metadata labels are exactly the saved labels.
    metadata_labels = metadata["label"].to_numpy().astype(np.int8)

    if not np.array_equal(metadata_labels, labels.astype(np.int8)):
        mismatches = np.sum(
            metadata_labels != labels.astype(np.int8)
        )

        raise AssertionError(
            f"LOMO-{fold}: {mismatches} label mismatches between "
            f"metadata and saved labels."
        )

    print(f"Windows : {len(probs):,}")
    print(f"Positive windows : {int(labels.sum()):,}")
    print(f"Metadata columns : {metadata.columns}")

    return metadata, probs.astype(np.float64), labels.astype(np.int8)


# ============================================================
# ONE SCORE PER DRIVE
# ============================================================

def make_drive_scores(
    metadata: pl.DataFrame,
    probs: np.ndarray,
):
    """
    Reduce window-level predictions to exactly one prediction
    score per SSD.

    Primary score:
        mean probability over the final 30-day window of that drive.

    The final 30 days means the 30 latest prediction windows available
    for that drive. If a drive has fewer than 30 windows, all available
    windows are used.

    The drive label is 1 if ANY window has label 1.
    """

    if metadata.height != len(probs):
        raise ValueError(
            "Metadata and probabilities must have identical length."
        )

    # Add probabilities to metadata.
    frame = metadata.with_columns(
        pl.Series(
            name="probability",
            values=probs,
        )
    )

    # Sort chronologically within each drive.
    frame = frame.sort(
        ["model", "disk_id", "window_date"]
    )

    # --------------------------------------------------------
    # Keep the final 30 prediction windows for every drive.
    # --------------------------------------------------------

    final_windows = (
        frame
        .group_by(
            ["model", "disk_id"],
            maintain_order=True,
        )
        .tail(30)
    )

    # --------------------------------------------------------
    # One row per drive.
    # --------------------------------------------------------

    drive_scores = (
        final_windows
        .group_by(
            ["model", "disk_id"],
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
        .sort(["model", "disk_id"])
    )

    # --------------------------------------------------------
    # Hard validation: exactly one row per drive.
    # --------------------------------------------------------

    duplicate_count = (
        drive_scores
        .group_by(["model", "disk_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_count != 0:
        raise AssertionError(
            "Drive-level aggregation produced duplicate drives."
        )

    if drive_scores.height == 0:
        raise ValueError("No drive-level scores were produced.")

    return drive_scores


# ============================================================
# DRIVE-LEVEL CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
):
    """
    Calculate drive-level ranking metrics.

    The threshold is NOT optimized here.
    For the primary analysis we use 0.5 as the probability threshold.
    """

    result = {}

    unique = np.unique(y_true)

    if len(unique) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y_true, scores)
        )

        result["pr_auc"] = float(
            average_precision_score(y_true, scores)
        )
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")

    predictions = (scores >= 0.5).astype(np.int8)

    result["precision"] = float(
        precision_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    result["recall"] = float(
        recall_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    return result


# ============================================================
# LAC CONFORMAL
# ============================================================

def lac_qhat(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    alpha: float,
):
    """
    Binary LAC conformal calibration.

    Score:
        y=1 -> 1-p
        y=0 -> p

    Finite-sample quantile:
        ceil((n+1)*(1-alpha))
    """

    scores = np.where(
        calibration_labels == 1,
        1.0 - calibration_scores,
        calibration_scores,
    )

    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:
        raise ValueError("Empty calibration set.")

    n = len(scores)

    rank = int(
        np.ceil((n + 1) * (1.0 - alpha))
    )

    rank = max(1, min(rank, n))

    sorted_scores = np.sort(scores)

    qhat = float(
        sorted_scores[rank - 1]
    )

    return qhat


def lac_predict_sets(
    scores: np.ndarray,
    qhat: float,
):
    """
    Construct binary LAC prediction sets.
    """

    p1 = scores
    p0 = 1.0 - scores

    include_0 = p0 <= qhat
    include_1 = p1 <= qhat

    return include_0, include_1


# ============================================================
# CONFORMAL REPORT
# ============================================================

def conformal_report(
    scores: np.ndarray,
    labels: np.ndarray,
    qhat: float,
):
    """
    Calculate overall, healthy-class and failure-class coverage.
    """

    include_0, include_1 = lac_predict_sets(
        scores,
        qhat,
    )

    covered = np.where(
        labels == 1,
        include_1,
        include_0,
    )

    set_size = (
        include_0.astype(np.int8)
        + include_1.astype(np.int8)
    )

    overall_coverage = float(
        np.mean(covered)
    )

    healthy_mask = labels == 0
    failure_mask = labels == 1

    if np.any(healthy_mask):
        healthy_coverage = float(
            np.mean(covered[healthy_mask])
        )
    else:
        healthy_coverage = float("nan")

    if np.any(failure_mask):
        failure_coverage = float(
            np.mean(covered[failure_mask])
        )
    else:
        failure_coverage = float("nan")

    return {
        "qhat": qhat,
        "coverage": overall_coverage,
        "coverage_healthy": healthy_coverage,
        "coverage_failure": failure_coverage,
        "avg_set_size": float(
            np.mean(set_size)
        ),
        "singleton_rate": float(
            np.mean(set_size == 1)
        ),
        "doubleton_rate": float(
            np.mean(set_size == 2)
        ),
        "empty_rate": float(
            np.mean(set_size == 0)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DRIVE-LEVEL CONFORMAL EVALUATION")
    print("=" * 70)

    all_rows = []
    all_drive_tables = []

    for fold in FOLDS:

        # ----------------------------------------------------
        # TEST DATA
        # ----------------------------------------------------

        test_metadata, test_probs, test_labels = load_fold(
            fold
        )

        # ----------------------------------------------------
        # DRIVE-LEVEL TEST SCORES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We still need CALIBRATION scores.
        #
        # These come from the existing LOMO calibration
        # predictions, using exactly the same drive-level
        # aggregation rule.
        # ----------------------------------------------------

        cal_prob_path = (
            RESULTS_DIR
            / f"lomo_{fold}_calibration_probs.npy"
        )

        cal_label_path = (
            RESULTS_DIR
            / f"lomo_{fold}_calibration_labels.npy"
        )

        cal_metadata_path = (
            WINDOWS_DIR
            / f"lomo_{fold}"
            / "calibration"
            / "metadata.parquet"
        )

        if not cal_prob_path.exists():
            raise FileNotFoundError(cal_prob_path)

        if not cal_label_path.exists():
            raise FileNotFoundError(cal_label_path)

        if not cal_metadata_path.exists():
            raise FileNotFoundError(cal_metadata_path)

        cal_probs = np.load(
            cal_prob_path
        ).astype(np.float64)

        cal_labels = np.load(
            cal_label_path
        ).astype(np.int8)

        cal_metadata = pl.read_parquet(
            cal_metadata_path
        )

        if len(cal_probs) != len(cal_labels):
            raise ValueError(
                f"LOMO-{fold}: calibration probability/label "
                f"length mismatch."
            )

        if cal_metadata.height != len(cal_probs):
            raise ValueError(
                f"LOMO-{fold}: calibration metadata/prediction "
                f"length mismatch."
            )

        metadata_cal_labels = (
            cal_metadata["label"]
            .to_numpy()
            .astype(np.int8)
        )

        if not np.array_equal(
            metadata_cal_labels,
            cal_labels,
        ):
            raise AssertionError(
                f"LOMO-{fold}: calibration label mismatch."
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

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if len(np.unique(cal_y)) < 2:
            raise ValueError(
                f"LOMO-{fold}: calibration drive set does not "
                f"contain both classes."
            )

        if len(np.unique(test_y)) < 2:
            raise ValueError(
                f"LOMO-{fold}: test drive set does not contain "
                f"both classes."
            )

        print()
        print(f"LOMO-{fold} drive counts:")
        print(f"  calibration drives : {len(cal_y):,}")
        print(f"  calibration failed : {int(cal_y.sum()):,}")
        print(f"  test drives        : {len(test_y):,}")
        print(f"  test failed        : {int(test_y.sum()):,}")

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        metrics = classification_metrics(
            test_y,
            test_scores,
        )

        # ----------------------------------------------------
        # LAC
        # ----------------------------------------------------

        qhat = lac_qhat(
            cal_scores,
            cal_y,
            ALPHA,
        )

        conformal = conformal_report(
            test_scores,
            test_y,
            qhat,
        )

        # ----------------------------------------------------
        # SAVE DRIVE-LEVEL TABLE
        # ----------------------------------------------------

        drive_path = (
            REPORTS_DIR
            / f"lomo_{fold}_drive_level_scores.csv"
        )

        test_drives.write_csv(
            drive_path
        )

        all_drive_tables.append(
            test_drives.with_columns(
                pl.lit(fold).alias("fold")
            )
        )

        # ----------------------------------------------------
        # RESULT ROW
        # ----------------------------------------------------

        row = {
            "fold": fold,

            "n_cal_drives": len(cal_y),
            "n_cal_failed_drives": int(cal_y.sum()),

            "n_test_drives": len(test_y),
            "n_test_failed_drives": int(test_y.sum()),

            "test_prevalence": float(
                np.mean(test_y)
            ),

            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],

            "qhat_lac": conformal["qhat"],
            "coverage_lac": conformal["coverage"],
            "coverage_healthy_lac": (
                conformal["coverage_healthy"]
            ),
            "coverage_failure_lac": (
                conformal["coverage_failure"]
            ),
            "avg_set_size_lac": (
                conformal["avg_set_size"]
            ),
            "singleton_rate_lac": (
                conformal["singleton_rate"]
            ),
            "doubleton_rate_lac": (
                conformal["doubleton_rate"]
            ),
            "empty_rate_lac": (
                conformal["empty_rate"]
            ),
        }

        all_rows.append(row)

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(f"LOMO-{fold} DRIVE-LEVEL RESULTS")
        print("-" * 70)

        print(
            f"ROC-AUC          : {metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC           : {metrics['pr_auc']:.4f}"
        )

        print(
            f"Precision        : {metrics['precision']:.4f}"
        )

        print(
            f"Recall           : {metrics['recall']:.4f}"
        )

        print(
            f"LAC qhat         : {conformal['qhat']:.6f}"
        )

        print(
            f"LAC coverage     : {conformal['coverage']:.4f}"
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
            f"Drive table saved: {drive_path}"
        )

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    result_df = pl.DataFrame(
        all_rows
    )

    result_path = (
        REPORTS_DIR
        / "drive_level_conformal_results.csv"
    )

    result_df.write_csv(
        result_path
    )

    # Combined drive tables.
    combined_drives = pl.concat(
        all_drive_tables,
        how="vertical",
    )

    combined_drive_path = (
        REPORTS_DIR
        / "lomo_all_drive_level_scores.csv"
    )

    combined_drives.write_csv(
        combined_drive_path
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DRIVE-LEVEL SUMMARY")
    print("=" * 70)

    print(
        result_df.select(
            [
                "fold",
                "n_test_drives",
                "n_test_failed_drives",
                "test_prevalence",
                "roc_auc",
                "pr_auc",
                "coverage_lac",
                "coverage_failure_lac",
                "avg_set_size_lac",
            ]
        )
    )

    print()
    print(
        f"Saved: {result_path}"
    )

    print(
        f"Saved: {combined_drive_path}"
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    main()