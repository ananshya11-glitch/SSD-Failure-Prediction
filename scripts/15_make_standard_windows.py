from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "processed" / "standard" / "telemetry"
OUTPUT_DIR = ROOT / "data" / "processed" / "standard" / "windows"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN = 30
STRIDE = 1
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


def make_windows(split_name):
    print()
    print("=" * 60)
    print(f"CREATING {split_name.upper()} WINDOWS")
    print("=" * 60)

    input_file = INPUT_DIR / f"{split_name}.parquet"

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    df = pl.read_parquet(input_file)

    print(f"Rows loaded: {df.height:,}")
    print(
        f"Drives: "
        f"{df.select(['model', 'disk_id']).unique().height:,}"
    )

    # --------------------------------------------------------
    # Sort by drive and date
    # --------------------------------------------------------

    df = df.sort(["model", "disk_id", "ds"])

    X_list = []
    y_list = []

    drive_count = 0
    window_count = 0

    # --------------------------------------------------------
    # Process one drive at a time
    # --------------------------------------------------------

    for (model, disk_id), drive in df.group_by(
        ["model", "disk_id"],
        maintain_order=True,
    ):

        drive_count += 1

        drive = drive.sort("ds")

        # ----------------------------------------------------
        # IMPORTANT:
        # Convert datetime -> date so it matches date_range().
        # ----------------------------------------------------

        dates = drive["ds"].dt.date()

        values = drive.select(COMMON_COLS).cast(
            pl.Float64
        )

        # ----------------------------------------------------
        # Create complete daily date range
        # ----------------------------------------------------

        start = dates.min()
        end = dates.max()

        if start is None or end is None:
            continue

        daily = pl.DataFrame(
            {
                "ds": pl.date_range(
                    start=start,
                    end=end,
                    interval="1d",
                    eager=True,
                )
            }
        )

        # ----------------------------------------------------
        # Join SMART values onto daily dates
        # ----------------------------------------------------

        telemetry = pl.DataFrame(
            {
                "ds": dates,
                **{
                    col: values[col]
                    for col in COMMON_COLS
                },
            }
        )

        daily = daily.join(
            telemetry,
            on="ds",
            how="left",
        )

        # ----------------------------------------------------
        # Calculate how much original data was available
        # before filling missing values.
        # ----------------------------------------------------

        valid_count = (
            daily
            .select(
                pl.all()
                .exclude("ds")
                .is_not_null()
                .cast(pl.Int8)
            )
            .sum_horizontal()
        )

        daily = daily.with_columns(
            (
                valid_count / len(COMMON_COLS)
            ).alias("_valid_frac")
        )

        # ----------------------------------------------------
        # Forward-fill missing SMART values.
        # ----------------------------------------------------

        daily = daily.with_columns(
            [
                pl.col(col).forward_fill()
                for col in COMMON_COLS
            ]
        )

        # ----------------------------------------------------
        # Leading missing values -> 0
        # ----------------------------------------------------

        daily = daily.with_columns(
            [
                pl.col(col).fill_null(0.0)
                for col in COMMON_COLS
            ]
        )

        # ----------------------------------------------------
        # Convert SMART data to NumPy
        # ----------------------------------------------------

        values_np = (
            daily
            .select(COMMON_COLS)
            .to_numpy()
            .astype(np.float32)
        )

        valid_np = daily["_valid_frac"].to_numpy()

        # ----------------------------------------------------
        # Get labels.
        #
        # Convert drive ds from datetime -> date so the join
        # key matches daily["ds"].
        # ----------------------------------------------------

        drive_labels = (
            drive
            .select(["ds", "label"])
            .with_columns(
                pl.col("ds").dt.date()
            )
        )

        labels = (
            daily
            .join(
                drive_labels,
                on="ds",
                how="left",
            )["label"]
            .fill_null(0)
            .to_numpy()
            .astype(np.int8)
        )

        # ----------------------------------------------------
        # Create 30-day windows
        # ----------------------------------------------------

        n_days = len(values_np)

        if n_days < WINDOW_LEN:
            continue

        for start_idx in range(
            0,
            n_days - WINDOW_LEN + 1,
            STRIDE,
        ):

            end_idx = start_idx + WINDOW_LEN

            # ------------------------------------------------
            # Every day in the window must have at least
            # MIN_VALID_FRAC original SMART observations.
            # ------------------------------------------------

            window_valid = valid_np[start_idx:end_idx]

            if np.mean(
                window_valid >= MIN_VALID_FRAC
            ) < 1.0:
                continue

            # ------------------------------------------------
            # Store window
            # ------------------------------------------------

            X_list.append(
                values_np[start_idx:end_idx]
            )

            # Label = final day of 30-day window
            y_list.append(
                labels[end_idx - 1]
            )

            window_count += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if drive_count % 500 == 0:
            print(
                f"Processed drives: {drive_count:,} | "
                f"windows: {window_count:,}"
            )

    # --------------------------------------------------------
    # Make NumPy arrays
    # --------------------------------------------------------

    if not X_list:
        raise RuntimeError(
            f"No windows were generated for {split_name}."
        )

    X = np.stack(X_list).astype(np.float32)

    y = np.asarray(
        y_list,
        dtype=np.int8,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    X_path = OUTPUT_DIR / f"{split_name}_X.npy"
    y_path = OUTPUT_DIR / f"{split_name}_y.npy"

    np.save(X_path, X)
    np.save(y_path, y)

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print(f"{split_name.upper()} COMPLETE")
    print(f"Shape: {X.shape}")
    print(f"Positives: {positives:,}")
    print(f"Negatives: {negatives:,}")
    print(
        f"Positive rate: "
        f"{positives / len(y):.4%}"
    )
    print(
        f"X size: "
        f"{X.nbytes / (1024 ** 3):.2f} GB"
    )
    print(f"Saved: {X_path}")
    print(f"Saved: {y_path}")


# ============================================================
# Run all three splits
# ============================================================

for split in [
    "train",
    "calibration",
    "test",
]:
    make_windows(split)


print()
print("=" * 60)
print("STANDARD WINDOW CREATION COMPLETE")
print("=" * 60)