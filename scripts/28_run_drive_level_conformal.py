from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "lomo"
WINDOWS_DIR = ROOT / "data" / "processed" / "lomo" / "windows"

REPORTS_DIR = ROOT / "reports"

FOLDS = ["A", "B", "C"]


# ============================================================
# CONFORMAL SETTINGS
# ============================================================

ALPHA = 0.10


# ============================================================
# HELPERS
# ============================================================

def load_array(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file:\n{path}"
        )

    return np.asarray(np.load(path)).reshape(-1)


def conformal_quantile(scores, alpha):
    """
    Finite-sample split-conformal quantile.

    qhat is the ceil((n+1)*(1-alpha))-th smallest
    calibration score.
    """

    scores = np.asarray(scores, dtype=float)

    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise ValueError("No finite calibration scores.")

    n = len(scores)

    rank = int(
        np.ceil((n + 1) * (1 - alpha))
    )

    rank = min(
        max(rank, 1),
        n
    )

    sorted_scores = np.sort(scores)

    return float(
        sorted_scores[rank - 1]
    )


def coverage_statistics(
    labels,
    include_0,
    include_1,
):
    """
    Calculate overall and class-conditional coverage.
    """

    labels = np.asarray(labels).astype(int)
    include_0 = np.asarray(include_0).astype(bool)
    include_1 = np.asarray(include_1).astype(bool)

    covered = np.where(
        labels == 1,
        include_1,
        include_0
    )

    overall = float(
        covered.mean()
    )

    healthy_mask = labels == 0
    failure_mask = labels == 1

    healthy_coverage = (
        float(covered[healthy_mask].mean())
        if healthy_mask.any()
        else np.nan
    )

    failure_coverage = (
        float(covered[failure_mask].mean())
        if failure_mask.any()
        else np.nan
    )

    set_size = (
        include_0.astype(int)
        + include_1.astype(int)
    )

    return {
        "coverage": overall,
        "healthy_coverage": healthy_coverage,
        "failure_coverage": failure_coverage,
        "avg_set_size": float(set_size.mean()),
        "singleton_rate": float((set_size == 1).mean()),
        "doubleton_rate": float((set_size == 2).mean()),
        "empty_rate": float((set_size == 0).mean()),
    }


# ============================================================
# BUILD ONE DRIVE-LEVEL DATASET
# ============================================================

def build_drive_level(
    fold,
    split,
):
    """
    Convert window-level probabilities into exactly
    one probability per drive.

    The selected score is the probability from the
    final available window for that drive.
    """

    metadata_path = (
        WINDOWS_DIR
        / f"lomo_{fold}"
        / split
        / "metadata.parquet"
    )

    if split == "calibration":
        probs_name = (
            f"lomo_{fold}_calibration_probs.npy"
        )

        labels_name = (
            f"lomo_{fold}_calibration_labels.npy"
        )

    elif split == "test":
        probs_name = (
            f"lomo_{fold}_test_probs.npy"
        )

        labels_name = (
            f"lomo_{fold}_test_labels.npy"
        )

    else:
        raise ValueError(
            f"Unknown split: {split}"
        )

    probs_path = RESULTS_DIR / probs_name
    labels_path = RESULTS_DIR / labels_name

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    metadata = pd.read_parquet(
        metadata_path
    )

    probs = load_array(
        probs_path
    )

    labels = load_array(
        labels_path
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(metadata) != len(probs):
        raise ValueError(
            f"{fold} {split}: metadata/probability mismatch: "
            f"{len(metadata):,} vs {len(probs):,}"
        )

    if len(metadata) != len(labels):
        raise ValueError(
            f"{fold} {split}: metadata/label mismatch: "
            f"{len(metadata):,} vs {len(labels):,}"
        )

    required = [
        "model",
        "disk_id",
        "window_date",
        "label",
    ]

    missing = [
        c for c in required
        if c not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"{fold} {split}: missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Construct dataframe
    # --------------------------------------------------------

    df = metadata.copy()

    df["probability"] = probs
    df["prediction_label"] = labels.astype(int)

    df["window_date"] = pd.to_datetime(
        df["window_date"]
    )

    df["drive_id"] = (
        df["model"].astype(str)
        + "_"
        + df["disk_id"].astype(str)
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    mismatch = (
        df["label"].astype(int)
        != df["prediction_label"].astype(int)
    ).sum()

    if mismatch != 0:
        raise ValueError(
            f"{fold} {split}: "
            f"{mismatch:,} label mismatches."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "drive_id",
            "window_date",
        ]
    )

    # --------------------------------------------------------
    # ONE SCORE PER DRIVE
    #
    # Last available prediction window.
    # --------------------------------------------------------

    drive_df = (
        df
        .groupby(
            "drive_id",
            sort=False
        )
        .tail(1)
        .copy()
    )

    drive_df = drive_df[
        [
            "drive_id",
            "model",
            "disk_id",
            "window_date",
            "probability",
            "label",
        ]
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    duplicate_drives = (
        drive_df["drive_id"]
        .duplicated()
        .sum()
    )

    if duplicate_drives != 0:
        raise ValueError(
            f"{fold} {split}: "
            f"{duplicate_drives} duplicate drive scores."
        )

    return drive_df


# ============================================================
# LAC
# ============================================================

def run_lac(
    cal_df,
    test_df,
):
    """
    Binary LAC conformal prediction.

    Score:
        y=1 -> 1-p
        y=0 -> p
    """

    cal_probs = cal_df["probability"].to_numpy()
    cal_labels = cal_df["label"].to_numpy().astype(int)

    test_probs = test_df["probability"].to_numpy()

    # --------------------------------------------------------
    # Calibration scores
    # --------------------------------------------------------

    cal_scores = np.where(
        cal_labels == 1,
        1.0 - cal_probs,
        cal_probs
    )

    qhat = conformal_quantile(
        cal_scores,
        ALPHA
    )

    # --------------------------------------------------------
    # Test prediction sets
    # --------------------------------------------------------

    test_include_1 = (
        1.0 - test_probs
        <= qhat
    )

    test_include_0 = (
        test_probs
        <= qhat
    )

    stats = coverage_statistics(
        test_df["label"].to_numpy(),
        test_include_0,
        test_include_1,
    )

    stats["method"] = "LAC"
    stats["qhat"] = qhat

    return stats


# ============================================================
# APS
# ============================================================

def run_aps(
    cal_df,
    test_df,
):
    """
    Binary APS conformal prediction.
    """

    cal_probs = cal_df["probability"].to_numpy()
    cal_labels = cal_df["label"].to_numpy().astype(int)

    test_probs = test_df["probability"].to_numpy()

    # --------------------------------------------------------
    # Calibration probabilities
    # --------------------------------------------------------

    cal_p1 = cal_probs
    cal_p0 = 1.0 - cal_probs

    cal_top_is_1 = (
        cal_p1 >= cal_p0
    )

    # Exact binary APS score
    cal_scores = np.where(
        cal_labels == 1,
        np.where(
            cal_top_is_1,
            cal_p1,
            cal_p0 + cal_p1
        ),
        np.where(
            cal_top_is_1,
            cal_p1 + cal_p0,
            cal_p0
        )
    )

    qhat = conformal_quantile(
        cal_scores,
        ALPHA
    )

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    test_p1 = test_probs
    test_p0 = 1.0 - test_probs

    test_top_is_1 = (
        test_p1 >= test_p0
    )

    # Exact binary APS inclusion
    include_1 = np.where(
        test_top_is_1,
        test_p1 <= qhat,
        1.0 <= qhat
    )

    include_0 = np.where(
        test_top_is_1,
        1.0 <= qhat,
        test_p0 <= qhat
    )

    stats = coverage_statistics(
        test_df["label"].to_numpy(),
        include_0,
        include_1,
    )

    stats["method"] = "APS"
    stats["qhat"] = qhat

    return stats


# ============================================================
# PRINT
# ============================================================

def print_result(
    fold,
    stats,
):
    print()
    print(
        f"{fold} — {stats['method']}"
    )
    print("-" * 75)

    print(
        f"qhat              : "
        f"{stats['qhat']:.6f}"
    )

    print(
        f"Overall coverage  : "
        f"{stats['coverage']:.6f}"
    )

    print(
        f"Healthy coverage  : "
        f"{stats['healthy_coverage']:.6f}"
    )

    print(
        f"Failure coverage  : "
        f"{stats['failure_coverage']:.6f}"
    )

    print(
        f"Average set size  : "
        f"{stats['avg_set_size']:.6f}"
    )

    print(
        f"Singleton rate    : "
        f"{stats['singleton_rate']:.6f}"
    )

    print(
        f"Doubleton rate    : "
        f"{stats['doubleton_rate']:.6f}"
    )

    print(
        f"Empty rate        : "
        f"{stats['empty_rate']:.6f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("DRIVE-LEVEL CONFORMAL ANALYSIS")
    print("=" * 75)

    print()
    print(
        "Drive score = probability from the "
        "last available window."
    )

    print(
        f"Target coverage = {1 - ALPHA:.2%}"
    )

    all_results = []

    for fold in FOLDS:

        print()
        print("=" * 75)
        print(f"LOMO-{fold}")
        print("=" * 75)

        # ----------------------------------------------------
        # Build drive-level calibration/test sets
        # ----------------------------------------------------

        cal_df = build_drive_level(
            fold,
            "calibration"
        )

        test_df = build_drive_level(
            fold,
            "test"
        )

        print()
        print("Drive-level sample sizes")
        print("-" * 75)

        print(
            f"Calibration drives : "
            f"{len(cal_df):,}"
        )

        print(
            f"Calibration failed : "
            f"{int(cal_df['label'].sum()):,}"
        )

        print(
            f"Test drives        : "
            f"{len(test_df):,}"
        )

        print(
            f"Test failed        : "
            f"{int(test_df['label'].sum()):,}"
        )

        # ----------------------------------------------------
        # LAC
        # ----------------------------------------------------

        lac = run_lac(
            cal_df,
            test_df
        )

        print_result(
            fold,
            lac
        )

        lac_record = {
            "fold": fold,
            **lac,
        }

        all_results.append(
            lac_record
        )

        # ----------------------------------------------------
        # APS
        # ----------------------------------------------------

        aps = run_aps(
            cal_df,
            test_df
        )

        print_result(
            fold,
            aps
        )

        aps_record = {
            "fold": fold,
            **aps,
        }

        all_results.append(
            aps_record
        )

        # ----------------------------------------------------
        # Save drive-level data
        # ----------------------------------------------------

        cal_output = (
            REPORTS_DIR
            / f"lomo_{fold}_drive_calibration.csv"
        )

        test_output = (
            REPORTS_DIR
            / f"lomo_{fold}_drive_test.csv"
        )

        cal_df.to_csv(
            cal_output,
            index=False
        )

        test_df.to_csv(
            test_output,
            index=False
        )

        print()
        print("Saved drive-level data:")
        print(cal_output)
        print(test_output)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        REPORTS_DIR
        / "drive_level_conformal_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 75)
    print("DRIVE-LEVEL CONFORMAL ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print("Results saved:")
    print(output_path)

    print()
    print(
        results_df[
            [
                "fold",
                "method",
                "qhat",
                "coverage",
                "healthy_coverage",
                "failure_coverage",
                "avg_set_size",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()