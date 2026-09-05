from pathlib import Path

import numpy as np
import polars as pl


# ============================================================
# PATHS / SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

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


# ============================================================
# LOMO METADATA
# ============================================================

def make_lomo_metadata(lomo, split_name):

    print()
    print("=" * 60)
    print(f"LOMO-{lomo} {split_name.upper()} METADATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Input / output paths
    # --------------------------------------------------------

    input_file = (
        ROOT
        / "data"
        / "processed"
        / "lomo"
        / "telemetry"
        / f"lomo_{lomo}"
        / f"{split_name}.parquet"
    )

    output_dir = (
        ROOT
        / "data"
        / "processed"
        / "lomo"
        / "windows"
        / f"lomo_{lomo}"
        / split_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load exactly the same input used by script 19
    # --------------------------------------------------------

    df = pl.read_parquet(input_file)

    print(f"Rows: {df.height:,}")
    print(
        "Drives: "
        f"{df.select(['model', 'disk_id']).unique().height:,}"
    )

    # EXACT SAME DATE CONVERSION AS SCRIPT 19
    df = df.with_columns(
        pl.col("ds")
        .dt.date()
        .alias("date")
    )

    # --------------------------------------------------------
    # Metadata lists
    # --------------------------------------------------------

    model_list = []
    disk_id_list = []
    date_list = []
    label_list = []

    # --------------------------------------------------------
    # EXACT SAME GROUPING AS SCRIPT 19
    # --------------------------------------------------------

    groups = df.partition_by(
        ["model", "disk_id"],
        maintain_order=True,
    )

    # --------------------------------------------------------
    # Process every drive
    # --------------------------------------------------------

    for drive_index, drive in enumerate(
        groups,
        start=1,
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        # Save identifiers BEFORE the select() below.
        # ----------------------------------------------------

        model = drive["model"][0]
        disk_id = drive["disk_id"][0]

        # EXACT SAME SORT AS SCRIPT 19
        drive = drive.sort("date")

        dates = drive["date"]

        if len(drive) < WINDOW_LEN:
            continue

        # ----------------------------------------------------
        # COMPLETE DAILY DATE RANGE
        # ----------------------------------------------------

        start = dates.min()
        end = dates.max()

        full_dates = pl.date_range(
            start,
            end,
            interval="1d",
            eager=True,
        )

        # ----------------------------------------------------
        # EXACT SAME DAILY RECONSTRUCTION AS SCRIPT 19
        # ----------------------------------------------------

        drive = (
            drive
            .select(
                ["date", "label"] + COMMON_COLS
            )
            .unique(
                subset=["date"],
                keep="last",
            )
            .join(
                pl.DataFrame(
                    {"date": full_dates}
                ),
                on="date",
                how="right",
            )
            .sort("date")
        )

        # ----------------------------------------------------
        # EXACT SAME LABEL HANDLING AS SCRIPT 19
        # ----------------------------------------------------

        drive = drive.with_columns(
            pl.col("label")
            .fill_null(0)
            .cast(pl.Int8)
        )

        # ----------------------------------------------------
        # EXACT SAME SMART FILL AS SCRIPT 19
        # ----------------------------------------------------

        drive = drive.with_columns(
            [
                pl.col(col)
                .forward_fill()
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias(col)
                for col in COMMON_COLS
            ]
        )

        # ----------------------------------------------------
        # Convert to arrays
        # ----------------------------------------------------

        labels = (
            drive["label"]
            .to_numpy()
        )

        dates_np = (
            drive["date"]
            .dt.strftime("%Y-%m-%d")
            .to_numpy()
        )

        n_days = len(dates_np)

        # ----------------------------------------------------
        # EXACT SAME WINDOW LOOP AS SCRIPT 19
        # ----------------------------------------------------

        for end_idx in range(
            WINDOW_LEN - 1,
            n_days,
        ):

            start_idx = (
                end_idx
                - WINDOW_LEN
                + 1
            )

            # ------------------------------------------------
            # EXACT SAME VALIDITY CHECK AS SCRIPT 19
            #
            # IMPORTANT:
            # This check occurs AFTER the SMART columns have
            # been forward-filled and zero-filled above.
            # ------------------------------------------------

            valid_slice = drive.slice(
                start_idx,
                WINDOW_LEN,
            )

            valid_count = (
                valid_slice
                .select(
                    [
                        pl.col(col)
                        .is_not_null()
                        for col in COMMON_COLS
                    ]
                )
                .to_numpy()
                .any(axis=1)
                .sum()
            )

            if (
                valid_count / WINDOW_LEN
                < MIN_VALID_FRAC
            ):
                continue

            # ------------------------------------------------
            # Store metadata for this exact window
            # ------------------------------------------------

            model_list.append(
                str(model)
            )

            disk_id_list.append(
                int(disk_id)
            )

            date_list.append(
                str(dates_np[end_idx])
            )

            label_list.append(
                int(labels[end_idx])
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if drive_index % 500 == 0:
            print(
                f"Processed drives: "
                f"{drive_index:,}/{len(groups):,} | "
                f"windows: {len(label_list):,}"
            )

    # ========================================================
    # CREATE METADATA DATAFRAME
    # ========================================================

    metadata = pl.DataFrame(
        {
            "model": model_list,
            "disk_id": disk_id_list,
            "window_date": date_list,
            "label": label_list,
        }
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        output_dir
        / "metadata.parquet"
    )

    metadata.write_parquet(
        output_file
    )

    # ========================================================
    # VERIFY AGAINST EXISTING y.npy
    # ========================================================

    y_file = (
        output_dir
        / "y.npy"
    )

    if not y_file.exists():
        raise FileNotFoundError(
            f"Existing y.npy not found:\n{y_file}"
        )

    existing_y = np.load(
        y_file,
        mmap_mode="r",
    )

    print()
    print("VERIFICATION")
    print("-" * 40)

    print(
        f"Metadata windows : "
        f"{len(metadata):,}"
    )

    print(
        f"Existing y.npy   : "
        f"{len(existing_y):,}"
    )

    # --------------------------------------------------------
    # Check 1: same number of windows
    # --------------------------------------------------------

    if len(metadata) != len(existing_y):

        raise RuntimeError(
            "\nWINDOW COUNT MISMATCH!\n"
            f"Metadata: {len(metadata):,}\n"
            f"y.npy:    {len(existing_y):,}"
        )

    print("✓ Window count matches")

    # --------------------------------------------------------
    # Check 2: same label ordering
    # --------------------------------------------------------

    metadata_labels = (
        metadata["label"]
        .to_numpy()
    )

    if not np.array_equal(
        metadata_labels,
        np.asarray(existing_y),
    ):

        mismatch = np.flatnonzero(
            metadata_labels
            != np.asarray(existing_y)
        )

        first_mismatch = (
            int(mismatch[0])
            if len(mismatch) > 0
            else -1
        )

        raise RuntimeError(
            "\nLABEL ORDER MISMATCH!\n"
            f"First mismatch: {first_mismatch}"
        )

    print("✓ Label order matches")

    # --------------------------------------------------------
    # Check 3: basic metadata sanity
    # --------------------------------------------------------

    if metadata["model"].null_count() > 0:
        raise RuntimeError(
            "Metadata contains null model values."
        )

    if metadata["disk_id"].null_count() > 0:
        raise RuntimeError(
            "Metadata contains null disk_id values."
        )

    if metadata["window_date"].null_count() > 0:
        raise RuntimeError(
            "Metadata contains null window dates."
        )

    print("✓ No missing drive metadata")

    print()
    print(
        f"✓ Saved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print("CREATING LOMO CONFORMAL WINDOW METADATA")
print("=" * 60)

for lomo in [
    "A",
    "B",
    "C",
]:

    for split in [
        "train",
        "calibration",
        "test",
    ]:

        make_lomo_metadata(
            lomo,
            split,
        )


print()
print("=" * 60)
print("LOMO CONFORMAL METADATA COMPLETE")
print("=" * 60)