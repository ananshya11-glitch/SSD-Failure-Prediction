from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT / "reports"

FOLDS = ["A", "B", "C"]


def summarize_fold(fold):
    path = REPORTS_DIR / f"lomo_{fold}_drive_test.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    df = pd.read_csv(path)

    required = [
        "drive_id",
        "model",
        "disk_id",
        "window_date",
        "probability",
        "label",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"LOMO-{fold}: missing columns: {missing}"
        )

    df["label"] = df["label"].astype(int)
    df["probability"] = df["probability"].astype(float)

    healthy = df[df["label"] == 0]["probability"]
    failed = df[df["label"] == 1]["probability"]

    if len(healthy) == 0 or len(failed) == 0:
        raise ValueError(
            f"LOMO-{fold}: missing healthy or failed drives."
        )

    print()
    print("=" * 75)
    print(f"LOMO-{fold} DRIVE-LEVEL SCORE ANALYSIS")
    print("=" * 75)

    print()
    print(f"Healthy drives : {len(healthy):,}")
    print(f"Failed drives  : {len(failed):,}")

    print()
    print("Probability statistics")
    print("-" * 75)

    rows = []

    for name, values in [
        ("Healthy", healthy),
        ("Failed", failed),
    ]:
        print()
        print(name)

        print(f"  Mean   : {values.mean():.6f}")
        print(f"  Median : {values.median():.6f}")
        print(f"  Std    : {values.std():.6f}")
        print(f"  Min    : {values.min():.6f}")
        print(f"  Max    : {values.max():.6f}")

        rows.append(
            {
                "fold": fold,
                "group": name,
                "n_drives": len(values),
                "mean_probability": values.mean(),
                "median_probability": values.median(),
                "std_probability": values.std(),
                "min_probability": values.min(),
                "max_probability": values.max(),
            }
        )

    # --------------------------------------------------------
    # Separation
    # --------------------------------------------------------

    mean_difference = (
        failed.mean() - healthy.mean()
    )

    print()
    print("Failed − Healthy mean probability")
    print("-" * 75)
    print(f"{mean_difference:.6f}")

    # --------------------------------------------------------
    # Fraction above useful thresholds
    # --------------------------------------------------------

    print()
    print("Fraction of drives above probability thresholds")
    print("-" * 75)

    threshold_rows = []

    for threshold in [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:

        healthy_fraction = (
            (healthy >= threshold).mean()
        )

        failed_fraction = (
            (failed >= threshold).mean()
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"healthy={healthy_fraction:.4f}, "
            f"failed={failed_fraction:.4f}"
        )

        threshold_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "healthy_fraction": healthy_fraction,
                "failed_fraction": failed_fraction,
            }
        )

    return rows, threshold_rows


def main():

    all_summary = []
    all_thresholds = []

    for fold in FOLDS:

        summary, thresholds = summarize_fold(
            fold
        )

        all_summary.extend(summary)
        all_thresholds.extend(thresholds)

    summary_df = pd.DataFrame(
        all_summary
    )

    threshold_df = pd.DataFrame(
        all_thresholds
    )

    summary_path = (
        REPORTS_DIR
        / "drive_level_score_summary.csv"
    )

    threshold_path = (
        REPORTS_DIR
        / "drive_level_score_thresholds.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    print()
    print("=" * 75)
    print("DRIVE-LEVEL SCORE ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print("Saved:")
    print(summary_path)
    print(threshold_path)

    print()
    print("Summary")
    print("-" * 75)

    print(
        summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()