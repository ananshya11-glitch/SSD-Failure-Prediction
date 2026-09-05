from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "lomo"
WINDOWS_DIR = ROOT / "data" / "processed" / "lomo" / "windows"

REPORTS_DIR = ROOT / "reports"


# ============================================================
# LOMO FOLDS
# ============================================================

FOLDS = ["A", "B", "C"]


# ============================================================
# LOAD NUMPY FILE
# ============================================================

def load_array(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file:\n{path}"
        )

    return np.asarray(np.load(path)).reshape(-1)


# ============================================================
# INSPECT ONE FOLD
# ============================================================

def inspect_fold(fold):

    print()
    print("=" * 75)
    print(f"LOMO-{fold}")
    print("=" * 75)

    # --------------------------------------------------------
    # Actual paths in your project
    # --------------------------------------------------------

    metadata_path = (
        WINDOWS_DIR
        / f"lomo_{fold}"
        / "test"
        / "metadata.parquet"
    )

    probs_path = (
        RESULTS_DIR
        / f"lomo_{fold}_test_probs.npy"
    )

    labels_path = (
        RESULTS_DIR
        / f"lomo_{fold}_test_labels.npy"
    )

    print()
    print("Files")
    print("-" * 75)
    print(f"Metadata : {metadata_path}")
    print(f"Probs    : {probs_path}")
    print(f"Labels   : {labels_path}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    metadata = pd.read_parquet(metadata_path)

    probs = load_array(probs_path)
    labels = load_array(labels_path)

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print()
    print("Basic checks")
    print("-" * 75)

    print(f"Metadata rows : {len(metadata):,}")
    print(f"Probabilities : {len(probs):,}")
    print(f"Labels        : {len(labels):,}")

    if len(metadata) != len(probs):
        raise ValueError(
            f"Metadata/probability mismatch in LOMO-{fold}: "
            f"{len(metadata):,} vs {len(probs):,}"
        )

    if len(metadata) != len(labels):
        raise ValueError(
            f"Metadata/label mismatch in LOMO-{fold}: "
            f"{len(metadata):,} vs {len(labels):,}"
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = [
        "model",
        "disk_id",
        "label",
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in LOMO-{fold} metadata: {missing}"
        )

    # --------------------------------------------------------
    # Attach predictions
    # --------------------------------------------------------

    metadata = metadata.copy()

    metadata["probability"] = probs
    metadata["prediction_label"] = labels.astype(int)

    # --------------------------------------------------------
    # Create drive identifier
    # --------------------------------------------------------

    metadata["drive_id"] = (
        metadata["model"].astype(str)
        + "_"
        + metadata["disk_id"].astype(str)
    )

    # --------------------------------------------------------
    # Verify labels match metadata
    # --------------------------------------------------------

    label_mismatch = (
        metadata["prediction_label"].astype(int)
        != metadata["label"].astype(int)
    ).sum()

    print(f"Label mismatches: {label_mismatch:,}")

    if label_mismatch != 0:
        raise ValueError(
            f"Prediction labels do not match metadata labels "
            f"in LOMO-{fold}."
        )

    # --------------------------------------------------------
    # Window statistics
    # --------------------------------------------------------

    print()
    print("Window statistics")
    print("-" * 75)

    total_windows = len(metadata)

    positive_windows = int(
        metadata["label"].sum()
    )

    negative_windows = total_windows - positive_windows

    print(f"Total windows   : {total_windows:,}")
    print(f"Positive windows: {positive_windows:,}")
    print(f"Negative windows: {negative_windows:,}")

    if total_windows:
        print(
            f"Positive rate   : "
            f"{100 * positive_windows / total_windows:.6f}%"
        )

    # --------------------------------------------------------
    # Drive-level aggregation
    # --------------------------------------------------------

    drive_summary = (
        metadata
        .groupby("drive_id", sort=False)
        .agg(
            model=("model", "first"),
            disk_id=("disk_id", "first"),
            windows=("drive_id", "size"),
            positive_windows=("label", "sum"),
            max_probability=("probability", "max"),
            mean_probability=("probability", "mean"),
            first_probability=("probability", "first"),
            last_probability=("probability", "last"),
            drive_label=("label", "max"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Drive counts
    # --------------------------------------------------------

    total_drives = len(drive_summary)

    failed_drives = int(
        (drive_summary["drive_label"] == 1).sum()
    )

    healthy_drives = int(
        (drive_summary["drive_label"] == 0).sum()
    )

    print()
    print("Drive statistics")
    print("-" * 75)

    print(f"Total drives   : {total_drives:,}")
    print(f"Healthy drives : {healthy_drives:,}")
    print(f"Failed drives  : {failed_drives:,}")

    if total_drives:
        print(
            f"Failure rate   : "
            f"{100 * failed_drives / total_drives:.6f}%"
        )

    # --------------------------------------------------------
    # Windows per drive
    # --------------------------------------------------------

    print()
    print("Windows per drive")
    print("-" * 75)

    print(
        f"Mean   : {drive_summary['windows'].mean():.2f}"
    )

    print(
        f"Median : {drive_summary['windows'].median():.2f}"
    )

    print(
        f"Min    : {drive_summary['windows'].min():,}"
    )

    print(
        f"Max    : {drive_summary['windows'].max():,}"
    )

    # --------------------------------------------------------
    # Failed-drive analysis
    # --------------------------------------------------------

    failed = drive_summary[
        drive_summary["drive_label"] == 1
    ].copy()

    healthy = drive_summary[
        drive_summary["drive_label"] == 0
    ].copy()

    print()
    print("Failed-drive analysis")
    print("-" * 75)

    if len(failed) > 0:

        positive_per_failed = (
            failed["positive_windows"]
        )

        print(
            f"Failed drives                  : "
            f"{len(failed):,}"
        )

        print(
            f"Mean windows/failed drive      : "
            f"{failed['windows'].mean():.2f}"
        )

        print(
            f"Mean positive windows         : "
            f"{positive_per_failed.mean():.2f}"
        )

        print(
            f"Median positive windows       : "
            f"{positive_per_failed.median():.2f}"
        )

        print(
            f"Maximum positive windows      : "
            f"{positive_per_failed.max():,}"
        )

        drives_with_positive = int(
            (positive_per_failed > 0).sum()
        )

        print(
            f"Failed drives with "
            f">0 positive windows          : "
            f"{drives_with_positive:,}"
        )

    # --------------------------------------------------------
    # Probability comparison
    # --------------------------------------------------------

    print()
    print("Drive-level probability comparison")
    print("-" * 75)

    if len(failed) > 0:

        print("FAILED DRIVES")

        print(
            f"  Max probability mean   : "
            f"{failed['max_probability'].mean():.6f}"
        )

        print(
            f"  Max probability median : "
            f"{failed['max_probability'].median():.6f}"
        )

        print(
            f"  Mean probability mean  : "
            f"{failed['mean_probability'].mean():.6f}"
        )

        print(
            f"  Mean probability median: "
            f"{failed['mean_probability'].median():.6f}"
        )

    if len(healthy) > 0:

        print()
        print("HEALTHY DRIVES")

        print(
            f"  Max probability mean   : "
            f"{healthy['max_probability'].mean():.6f}"
        )

        print(
            f"  Max probability median : "
            f"{healthy['max_probability'].median():.6f}"
        )

        print(
            f"  Mean probability mean  : "
            f"{healthy['mean_probability'].mean():.6f}"
        )

        print(
            f"  Mean probability median: "
            f"{healthy['mean_probability'].median():.6f}"
        )

    # --------------------------------------------------------
    # Save drive-level table
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        REPORTS_DIR
        / f"lomo_{fold}_drive_level_inspection.csv"
    )

    drive_summary.to_csv(
        output_path,
        index=False
    )

    print()
    print(f"Saved:")
    print(output_path)

    return drive_summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("DRIVE-LEVEL LOMO INSPECTION")
    print("=" * 75)

    print()
    print(f"Results directory : {RESULTS_DIR}")
    print(f"Windows directory : {WINDOWS_DIR}")

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory does not exist:\n{RESULTS_DIR}"
        )

    if not WINDOWS_DIR.exists():
        raise FileNotFoundError(
            f"Windows directory does not exist:\n{WINDOWS_DIR}"
        )

    summaries = []

    for fold in FOLDS:

        summary = inspect_fold(fold)

        summary["fold"] = fold

        summaries.append(summary)

    # --------------------------------------------------------
    # Combined result
    # --------------------------------------------------------

    combined = pd.concat(
        summaries,
        ignore_index=True
    )

    combined_path = (
        REPORTS_DIR
        / "lomo_all_drive_level_inspection.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    # --------------------------------------------------------
    # Final overview
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("ALL LOMO FOLDS COMPLETED")
    print("=" * 75)

    print()
    print("Combined file:")
    print(combined_path)

    print()
    print("Overview")
    print("-" * 75)

    overview = (
        combined
        .groupby("fold")
        .agg(
            drives=("drive_id", "size"),
            failed_drives=("drive_label", "sum"),
            mean_windows=("windows", "mean"),
            mean_positive_windows=(
                "positive_windows",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        overview.to_string(index=False)
    )


if __name__ == "__main__":
    main()