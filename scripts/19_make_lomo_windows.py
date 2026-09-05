from pathlib import Path

import numpy as np
import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("data/processed/lomo/telemetry")
OUT = Path("data/processed/lomo/windows")

WINDOW_LEN = 30

# Sample prediction dates every 30 days.
STRIDE = 30

# No positive oversampling.
POSITIVE_STRIDE = 1

# Minimum fraction of originally observed SMART cells
# required inside a 30-day window.
MIN_VALID_FRAC = 0.5


# Common SMART features used by all three vendors.
COMMON_COLS = [
    "n_5",
    "r_5",
    "n_9",
    "r_9",
    "n_12",
    "r_12",
    "n_183",
    "r_183",
    "n_184",
    "r_184",
    "n_187",
    "r_187",
    "n_197",
    "r_197",
    "n_199",
    "r_199"
]


LOMO_CONFIG = {
    "A": ["train", "calibration", "test"],
    "B": ["train", "calibration", "test"],
    "C": ["train", "calibration", "test"],
}


# ============================================================
# MAKE WINDOWS FOR ONE LOMO SPLIT
# ============================================================

def make_windows(parquet_path: Path, out_dir: Path):

    print()
    print("=" * 70)
    print(f"PROCESSING: {parquet_path}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    df = pl.read_parquet(parquet_path)

    required_cols = [
        "model",
        "disk_id",
        "ds",
        "label",
        *COMMON_COLS,
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Rows  : {df.height:,}")

    print(
        "Drives:",
        df.select(
            ["model", "disk_id"]
        ).unique().height,
    )

    # --------------------------------------------------------
    # Convert ds to calendar date.
    #
    # The labels are defined using calendar days, so the
    # window date must also be a calendar date.
    # --------------------------------------------------------

    df = df.with_columns(
        pl.col("ds")
        .dt.date()
        .alias("date")
    )

    # --------------------------------------------------------
    # Storage for resulting windows
    # --------------------------------------------------------

    X_list = []
    y_list = []

    metadata_model = []
    metadata_disk = []
    metadata_date = []
    metadata_label = []

    # --------------------------------------------------------
    # Process one drive at a time.
    # --------------------------------------------------------

    groups = df.partition_by(
        ["model", "disk_id"],
        maintain_order=True,
    )

    total_drives = len(groups)

    print(
        f"Processing {total_drives:,} drives..."
    )

    for drive_number, drive in enumerate(
        groups,
        start=1,
    ):

        # ----------------------------------------------------
        # Sort chronologically.
        # ----------------------------------------------------

        drive = drive.sort("date")

        model = drive["model"][0]
        disk_id = drive["disk_id"][0]

        # ----------------------------------------------------
        # Original observed dates.
        # ----------------------------------------------------

        observed_dates = drive["date"]

        if len(observed_dates) == 0:
            continue

        first_date = observed_dates.min()
        last_date = observed_dates.max()

        # ----------------------------------------------------
        # Build complete daily calendar.
        # ----------------------------------------------------

        full_dates = pl.date_range(
            first_date,
            last_date,
            interval="1d",
            eager=True,
        )

        # ----------------------------------------------------
        # Keep only fields needed for windowing.
        #
        # If there is accidentally more than one row for a
        # drive/day, keep the last one.
        # ----------------------------------------------------

        drive = (
            drive
            .select(
                [
                    "date",
                    "label",
                    *COMMON_COLS,
                ]
            )
            .unique(
                subset=["date"],
                keep="last",
            )
            .join(
                pl.DataFrame(
                    {
                        "date": full_dates
                    }
                ),
                on="date",
                how="right",
            )
            .sort("date")
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT replace missing labels with 0.
        #
        # A missing label means that the label for that
        # calendar day is undefined.
        #
        # We represent it internally as -1 and skip windows
        # ending on such days.
        # ----------------------------------------------------

        labels = (
            drive["label"]
            .fill_null(-1)
            .cast(pl.Int8)
            .to_numpy()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Calculate SMART validity BEFORE imputation.
        #
        # This is a cell-level validity matrix:
        #
        #   True  = original SMART value existed
        #   False = SMART value was missing
        #
        # Shape:
        #
        #   number_of_days x 16_features
        #
        # The 30-day window is valid when the fraction of
        # originally present SMART cells is >= 0.5.
        # ----------------------------------------------------

        original_smart = (
            drive
            .select(COMMON_COLS)
            .to_numpy()
        )

        # ----------------------------------------------------
        # Convert to boolean presence matrix.
        #
        # Numeric SMART values can contain NaN, so handle both
        # Polars nulls and NumPy NaNs safely.
        # ----------------------------------------------------

        smart_present = ~np.isnan(
            original_smart.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Now perform forward-fill.
        #
        # IMPORTANT:
        # validity continues to use smart_present above.
        # We never calculate validity from these filled values.
        # ----------------------------------------------------

        drive = drive.with_columns(
            [
                (
                    pl.col(c)
                    .forward_fill()
                    .fill_null(0.0)
                    .cast(pl.Float32)
                    .alias(c)
                )
                for c in COMMON_COLS
            ]
        )

        # ----------------------------------------------------
        # Convert filled SMART values to NumPy.
        # ----------------------------------------------------

        values = (
            drive
            .select(COMMON_COLS)
            .to_numpy()
            .astype(np.float32)
        )

        n_days = values.shape[0]

        # ----------------------------------------------------
        # After reindexing, determine whether the drive has
        # enough calendar days for a 30-day window.
        # ----------------------------------------------------

        if n_days < WINDOW_LEN:
            continue

        # ----------------------------------------------------
        # Sliding windows.
        #
        # Window ending at T contains:
        #
        #   T-29, ..., T-1, T
        #
        # and its label is the label at T.
        # ----------------------------------------------------

        first_end = WINDOW_LEN - 1

        for end_idx in range(
            first_end,
            n_days,
            STRIDE,
        ):

            # ------------------------------------------------
            # Label of the final day.
            # ------------------------------------------------

            label = int(
                labels[end_idx]
            )

            # ------------------------------------------------
            # Undefined label:
            # skip this prediction date.
            # ------------------------------------------------

            if label < 0:
                continue

            # ------------------------------------------------
            # Positive stride.
            #
            # POSITIVE_STRIDE = 1, so every positive prediction
            # date is retained.
            # ------------------------------------------------

            if label == 1:

                if (
                    (end_idx - first_end)
                    % POSITIVE_STRIDE
                    != 0
                ):
                    continue

            # ------------------------------------------------
            # Window boundaries.
            # ------------------------------------------------

            start_idx = (
                end_idx
                - WINDOW_LEN
                + 1
            )

            # ------------------------------------------------
            # Original SMART validity.
            #
            # This is deliberately calculated from
            # smart_present, NOT from the forward-filled data.
            # ------------------------------------------------

            valid_slice = smart_present[
                start_idx:
                end_idx + 1
            ]

            valid_frac = float(
                valid_slice.mean()
            )

            if valid_frac < MIN_VALID_FRAC:
                continue

            # ------------------------------------------------
            # Store the window.
            # ------------------------------------------------

            X_list.append(
                values[
                    start_idx:
                    end_idx + 1
                ]
            )

            y_list.append(label)

            # ------------------------------------------------
            # Metadata is stored in exactly the same order as
            # X and y.
            # ------------------------------------------------

            metadata_model.append(model)

            metadata_disk.append(disk_id)

            metadata_date.append(
                drive["date"][end_idx]
            )

            metadata_label.append(label)

        # ----------------------------------------------------
        # Progress indicator.
        # ----------------------------------------------------

        if (
            drive_number % 500 == 0
            or drive_number == total_drives
        ):
            print(
                f"Processed drives: "
                f"{drive_number:,}/{total_drives:,}"
            )

    # ========================================================
    # BUILD FINAL ARRAYS
    # ========================================================

    if len(X_list) == 0:

        X = np.empty(
            (
                0,
                WINDOW_LEN,
                len(COMMON_COLS),
            ),
            dtype=np.float32,
        )

        y = np.empty(
            0,
            dtype=np.int8,
        )

        metadata = pl.DataFrame(
            {
                "model": pl.Series(
                    [],
                    dtype=pl.String,
                ),
                "disk_id": pl.Series(
                    [],
                    dtype=pl.Int64,
                ),
                "window_date": pl.Series(
                    [],
                    dtype=pl.Date,
                ),
                "label": pl.Series(
                    [],
                    dtype=pl.Int8,
                ),
            }
        )

    else:

        X = np.asarray(
            X_list,
            dtype=np.float32,
        )

        y = np.asarray(
            y_list,
            dtype=np.int8,
        )

        metadata = pl.DataFrame(
            {
                "model": metadata_model,
                "disk_id": metadata_disk,
                "window_date": metadata_date,
                "label": metadata_label,
            }
        )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    print()
    print("Running sanity checks...")

    # X and y must have the same number of samples.
    if X.shape[0] != y.shape[0]:
        raise AssertionError(
            f"X/y mismatch: "
            f"{X.shape[0]} vs {y.shape[0]}"
        )

    # Metadata must have exactly one row per window.
    if metadata.height != y.shape[0]:
        raise AssertionError(
            f"Metadata/y mismatch: "
            f"{metadata.height} vs {y.shape[0]}"
        )

    # Metadata labels must exactly equal y.
    metadata_y = (
        metadata["label"]
        .to_numpy()
        .astype(np.int8)
    )

    if not np.array_equal(
        metadata_y,
        y,
    ):
        raise AssertionError(
            "Metadata labels do not match y."
        )

    # Labels must only be 0 or 1.
    unique_labels = np.unique(y)

    if not set(
        unique_labels.tolist()
    ).issubset({0, 1}):

        raise AssertionError(
            f"Unexpected labels: "
            f"{unique_labels}"
        )

    # Correct shape.
    if X.ndim != 3:
        raise AssertionError(
            f"Expected X to have 3 dimensions, "
            f"got {X.ndim}"
        )

    if X.shape[1] != WINDOW_LEN:
        raise AssertionError(
            f"Expected window length "
            f"{WINDOW_LEN}, got {X.shape[1]}"
        )

    if X.shape[2] != len(COMMON_COLS):
        raise AssertionError(
            f"Expected {len(COMMON_COLS)} features, "
            f"got {X.shape[2]}"
        )

    # No NaNs should remain after forward/zero filling.
    if np.isnan(X).any():
        raise AssertionError(
            "X contains NaN values after imputation."
        )

    # ========================================================
    # SAVE
    # ========================================================

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_path = out_dir / "X.npy"
    y_path = out_dir / "y.npy"
    metadata_path = out_dir / "metadata.parquet"

    np.save(
        X_path,
        X,
    )

    np.save(
        y_path,
        y,
    )

    metadata.write_parquet(
        metadata_path
    )

    # ========================================================
    # REPORT
    # ========================================================

    positives = int(
        (y == 1).sum()
    )

    negatives = int(
        (y == 0).sum()
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"X shape          : {X.shape}"
    )

    print(
        f"X size           : "
        f"{X.nbytes / (1024 ** 3):.2f} GB"
    )

    print(
        f"Total windows    : "
        f"{len(y):,}"
    )

    print(
        f"Positive windows : "
        f"{positives:,}"
    )

    print(
        f"Negative windows : "
        f"{negatives:,}"
    )

    if len(y) > 0:
        print(
            f"Positive rate    : "
            f"{y.mean():.6%}"
        )
    else:
        print(
            "Positive rate    : 0.000000%"
        )

    print(
        f"Metadata rows    : "
        f"{metadata.height:,}"
    )

    print()
    print("Saved:")
    print(f"  {X_path}")
    print(f"  {y_path}")
    print(f"  {metadata_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for lomo, splits in LOMO_CONFIG.items():

        for split in splits:

            parquet_path = (
                ROOT
                / f"lomo_{lomo}"
                / f"{split}.parquet"
            )

            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Telemetry file not found:\n"
                    f"{parquet_path}"
                )

            out_dir = (
                OUT
                / f"lomo_{lomo}"
                / split
            )

            make_windows(
                parquet_path,
                out_dir,
            )

    print()
    print("=" * 70)
    print("ALL CORRECTED LOMO WINDOWS CREATED")
    print("=" * 70)