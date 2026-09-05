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
# LOAD NUMPY ARRAY
# ============================================================

def load_array(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file:\n{path}"
        )

    return np.asarray(np.load(path)).reshape(-1)


# ============================================================
# ANALYZE ONE FOLD
# ============================================================

def analyze_fold(fold):

    print()
    print("=" * 75)
    print(f"LOMO-{fold} — FAILURE-HORIZON ANALYSIS")
    print("=" * 75)

    # --------------------------------------------------------
    # ACTUAL PATHS
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

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    metadata = pd.read_parquet(metadata_path)

    probs = load_array(probs_path)
    labels = load_array(labels_path)

    # --------------------------------------------------------
    # CHECK LENGTHS
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
    # ACTUAL METADATA COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "model",
        "disk_id",
        "window_date",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in LOMO-{fold}: {missing}"
        )

    # --------------------------------------------------------
    # ATTACH PREDICTIONS
    # --------------------------------------------------------

    df = metadata.copy()

    df["probability"] = probs
    df["prediction_label"] = labels.astype(int)

    # Drive identifier
    df["drive_id"] = (
        df["model"].astype(str)
        + "_"
        + df["disk_id"].astype(str)
    )

    # Correct actual date column
    df["window_date"] = pd.to_datetime(
        df["window_date"]
    )

    # Sort chronologically within each drive
    df = df.sort_values(
        ["drive_id", "window_date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # LABEL ALIGNMENT CHECK
    # --------------------------------------------------------

    label_mismatch = (
        df["label"].astype(int)
        != df["prediction_label"].astype(int)
    ).sum()

    print(f"Label mismatches: {label_mismatch:,}")

    if label_mismatch != 0:
        raise ValueError(
            f"Prediction labels do not match metadata labels "
            f"in LOMO-{fold}."
        )

    # --------------------------------------------------------
    # DRIVE-LEVEL ANALYSIS
    # --------------------------------------------------------

    records = []

    for drive_id, drive in df.groupby(
        "drive_id",
        sort=False
    ):

        drive = drive.sort_values(
            "window_date"
        )

        drive_label = int(
            drive["label"].max()
        )

        positive = drive[
            drive["label"] == 1
        ].copy()

        record = {
            "fold": fold,
            "drive_id": drive_id,
            "model": drive["model"].iloc[0],
            "disk_id": drive["disk_id"].iloc[0],

            "drive_label": drive_label,

            "total_windows": len(drive),
            "positive_windows": len(positive),

            "first_window_date": (
                drive["window_date"].min()
            ),

            "last_window_date": (
                drive["window_date"].max()
            ),
        }

        # ----------------------------------------------------
        # HEALTHY DRIVE
        # ----------------------------------------------------

        if drive_label == 0:

            record.update({
                "first_positive_date": pd.NaT,
                "last_positive_date": pd.NaT,

                "first_positive_probability": np.nan,
                "last_positive_probability": np.nan,
                "max_positive_probability": np.nan,

                "max_probability": (
                    drive["probability"].max()
                ),

                "mean_probability": (
                    drive["probability"].mean()
                ),

                "last_window_probability": (
                    drive["probability"].iloc[-1]
                ),
            })

        # ----------------------------------------------------
        # FAILED DRIVE
        # ----------------------------------------------------

        else:

            if len(positive) == 0:
                raise ValueError(
                    f"Failed drive {drive_id} in LOMO-{fold} "
                    f"has zero positive windows."
                )

            first_positive = positive.iloc[0]
            last_positive = positive.iloc[-1]

            record.update({
                "first_positive_date": (
                    first_positive["window_date"]
                ),

                "last_positive_date": (
                    last_positive["window_date"]
                ),

                "first_positive_probability": (
                    first_positive["probability"]
                ),

                "last_positive_probability": (
                    last_positive["probability"]
                ),

                "max_positive_probability": (
                    positive["probability"].max()
                ),

                "max_probability": (
                    drive["probability"].max()
                ),

                "mean_probability": (
                    drive["probability"].mean()
                ),

                "last_window_probability": (
                    drive["probability"].iloc[-1]
                ),
            })

        records.append(record)

    result = pd.DataFrame(records)

    # ========================================================
    # SPLIT FAILED / HEALTHY
    # ========================================================

    failed = result[
        result["drive_label"] == 1
    ].copy()

    healthy = result[
        result["drive_label"] == 0
    ].copy()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("Drive counts")
    print("-" * 75)

    print(f"Total drives   : {len(result):,}")
    print(f"Failed drives  : {len(failed):,}")
    print(f"Healthy drives : {len(healthy):,}")

    # --------------------------------------------------------
    # POSITIVE WINDOW STRUCTURE
    # --------------------------------------------------------

    print()
    print("Positive-window structure")
    print("-" * 75)

    if len(failed) > 0:

        print(
            f"Mean positive windows/failed drive   : "
            f"{failed['positive_windows'].mean():.2f}"
        )

        print(
            f"Median positive windows/failed drive : "
            f"{failed['positive_windows'].median():.2f}"
        )

        print(
            f"Minimum positive windows             : "
            f"{failed['positive_windows'].min():.0f}"
        )

        print(
            f"Maximum positive windows             : "
            f"{failed['positive_windows'].max():.0f}"
        )

    # --------------------------------------------------------
    # FAILURE HORIZON PROBABILITIES
    # --------------------------------------------------------

    print()
    print("CNN probability during failure horizon")
    print("-" * 75)

    if len(failed) > 0:

        print("FAILED DRIVES")
        print()

        print(
            f"First positive-window probability mean   : "
            f"{failed['first_positive_probability'].mean():.6f}"
        )

        print(
            f"First positive-window probability median : "
            f"{failed['first_positive_probability'].median():.6f}"
        )

        print()

        print(
            f"Last positive-window probability mean    : "
            f"{failed['last_positive_probability'].mean():.6f}"
        )

        print(
            f"Last positive-window probability median  : "
            f"{failed['last_positive_probability'].median():.6f}"
        )

        print()

        print(
            f"Maximum probability during horizon mean   : "
            f"{failed['max_positive_probability'].mean():.6f}"
        )

        print(
            f"Maximum probability during horizon median : "
            f"{failed['max_positive_probability'].median():.6f}"
        )

    # --------------------------------------------------------
    # HEALTHY COMPARISON
    # --------------------------------------------------------

    print()
    print("Healthy-drive comparison")
    print("-" * 75)

    if len(healthy) > 0:

        print(
            f"Healthy max probability mean     : "
            f"{healthy['max_probability'].mean():.6f}"
        )

        print(
            f"Healthy max probability median   : "
            f"{healthy['max_probability'].median():.6f}"
        )

        print(
            f"Healthy mean probability mean    : "
            f"{healthy['mean_probability'].mean():.6f}"
        )

        print(
            f"Healthy mean probability median  : "
            f"{healthy['mean_probability'].median():.6f}"
        )

        print(
            f"Healthy last-window probability mean : "
            f"{healthy['last_window_probability'].mean():.6f}"
        )

        print(
            f"Healthy last-window probability median : "
            f"{healthy['last_window_probability'].median():.6f}"
        )

    # --------------------------------------------------------
    # FAILURE-HORIZON DURATION
    # --------------------------------------------------------

    print()
    print("Failure-horizon duration")
    print("-" * 75)

    if len(failed) > 0:

        duration = (
            failed["last_positive_date"]
            - failed["first_positive_date"]
        ).dt.days

        print(
            f"Mean days between first/last positive : "
            f"{duration.mean():.2f}"
        )

        print(
            f"Median days between first/last positive : "
            f"{duration.median():.2f}"
        )

        print(
            f"Minimum days : {duration.min():.0f}"
        )

        print(
            f"Maximum days : {duration.max():.0f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        REPORTS_DIR
        / f"lomo_{fold}_failure_horizon.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print()
    print("Saved:")
    print(output_path)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("FAILURE-HORIZON ANALYSIS")
    print("=" * 75)

    all_results = []

    for fold in FOLDS:

        result = analyze_fold(fold)

        all_results.append(result)

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined_path = (
        REPORTS_DIR
        / "lomo_all_failure_horizon.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print()
    print("=" * 75)
    print("FAILURE-HORIZON ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print("Combined result:")
    print(combined_path)


if __name__ == "__main__":
    main()