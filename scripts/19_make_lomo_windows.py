from pathlib import Path
import numpy as np
import polars as pl

# ============================================================
# CONFIG
# ============================================================

ROOT = Path("data/processed/lomo/telemetry")
OUT = Path("data/processed/lomo/windows")

WINDOW_LEN = 30
MIN_VALID_FRAC = 0.5

COMMON_COLS = [
    "n_5", "r_5",
    "n_9", "r_9",
    "n_12", "r_12",
    "n_183", "r_183",
    "n_184", "r_184",
    "n_187", "r_187",
    "n_197", "r_197",
    "n_199", "r_199",
]

LOMO_CONFIG = {
    "A": ["train", "calibration", "test"],
    "B": ["train", "calibration", "test"],
    "C": ["train", "calibration", "test"],
}


# ============================================================
# WINDOW CREATION
# ============================================================

def make_windows(parquet_path, out_dir):
    print("\n" + "=" * 60)
    print(f"PROCESSING: {parquet_path}")
    print("=" * 60)

    df = pl.read_parquet(parquet_path)

    print(f"Rows: {df.height}")
    print(f"Drives: {df.select(['model', 'disk_id']).unique().height}")

    # Make sure dates are actual calendar dates
    df = df.with_columns(
        pl.col("ds").dt.date().alias("date")
    )

    X_list = []
    y_list = []

    # Process drive-by-drive
    groups = df.partition_by(
        ["model", "disk_id"],
        maintain_order=True
    )

    for i, drive in enumerate(groups, start=1):

        drive = drive.sort("date")

        dates = drive["date"]

        if len(drive) < WINDOW_LEN:
            continue

        # Reindex onto complete daily range
        start = dates.min()
        end = dates.max()

        full_dates = pl.date_range(
            start,
            end,
            interval="1d",
            eager=True
        )

        drive = (
            drive
            .select(["date", "label"] + COMMON_COLS)
            .unique(subset=["date"], keep="last")
            .join(
                pl.DataFrame({"date": full_dates}),
                on="date",
                how="right"
            )
            .sort("date")
        )

        # Preserve labels
        drive = drive.with_columns(
            pl.col("label").fill_null(0).cast(pl.Int8)
        )

        # Check valid SMART fraction before filling
        smart_valid = (
            pl.any_horizontal(
                [pl.col(c).is_not_null() for c in COMMON_COLS]
            )
        )

        # Fill SMART values
        drive = drive.with_columns([
            pl.col(c)
            .forward_fill()
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(c)
            for c in COMMON_COLS
        ])

        values = drive.select(COMMON_COLS).to_numpy()
        labels = drive["label"].to_numpy()

        # ----------------------------------------------------
        # Sliding windows
        # ----------------------------------------------------

        for end_idx in range(WINDOW_LEN - 1, len(values)):

            start_idx = end_idx - WINDOW_LEN + 1

            # Original rows that contributed to this window
            valid_slice = drive.slice(
                start_idx,
                WINDOW_LEN
            )

            valid_count = (
                valid_slice
                .select([
                    pl.col(c).is_not_null()
                    for c in COMMON_COLS
                ])
                .to_numpy()
                .any(axis=1)
                .sum()
            )

            if valid_count / WINDOW_LEN < MIN_VALID_FRAC:
                continue

            X_list.append(
                values[start_idx:end_idx + 1]
            )

            # Label is the final day of the window
            y_list.append(
                labels[end_idx]
            )

        if i % 500 == 0:
            print(f"Processed drives: {i}/{len(groups)}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    print("\nRESULT")
    print("-" * 40)
    print(f"X shape: {X.shape}")
    print(f"X size: {X.nbytes / (1024**3):.2f} GB")
    print(f"Positive windows: {int(y.sum()):,}")
    print(f"Negative windows: {int((y == 0).sum()):,}")
    print(f"Positive rate: {y.mean():.6%}")

    print(f"\nSaved:")
    print(out_dir / "X.npy")
    print(out_dir / "y.npy")


# ============================================================
# MAIN
# ============================================================

for lomo, splits in LOMO_CONFIG.items():

    for split in splits:

        parquet_path = (
            ROOT /
            f"lomo_{lomo}" /
            f"{split}.parquet"
        )

        out_dir = (
            OUT /
            f"lomo_{lomo}" /
            split
        )

        make_windows(
            parquet_path,
            out_dir
        )

print("\n" + "=" * 60)
print("ALL LOMO WINDOWS CREATED")
print("=" * 60)