from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

LABELLED = ROOT / "data" / "processed" / "alibaba_ssd_labelled.parquet"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "data" / "processed" / "windows"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN = 30
STRIDE = 1
MIN_VALID_FRAC = 0.5

DRIVES_PER_BATCH = 500

FEATURES = [
    "n_5", "r_5",
    "n_9", "r_9",
    "n_12", "r_12",
    "n_183", "r_183",
    "n_184", "r_184",
    "n_187", "r_187",
    "n_197", "r_197",
    "n_199", "r_199",
]


# ---------------------------------------------------------------
# Create windows for one drive
# ---------------------------------------------------------------

def make_drive_windows(g):

    g = g.sort("ds")

    first = g["ds"].min()
    last = g["ds"].max()

    dates = pl.datetime_range(
        first,
        last,
        interval="1d",
        eager=True,
    )

    j = (
        pl.DataFrame({"ds": dates})
        .join(g, on="ds", how="left")
        .sort("ds")
    )

    values = (
        j.select(FEATURES)
        .to_numpy()
        .astype(np.float32)
    )

    # Original-value mask.
    present = np.isfinite(values)

    # Forward-fill missing values.
    for c in range(values.shape[1]):

        last_valid = -1

        for i in range(len(values)):

            if np.isfinite(values[i, c]):
                last_valid = i

            elif last_valid >= 0:
                values[i, c] = values[last_valid, c]

            else:
                values[i, c] = 0.0

    labels = (
        j["label"]
        .fill_null(-1)
        .to_numpy()
        .astype(np.int8)
    )

    end_dates = j["ds"].to_list()

    X = []
    y = []
    ends = []

    for end in range(
        WINDOW_LEN - 1,
        len(values),
        STRIDE,
    ):

        # End day must have an observed label.
        if labels[end] < 0:
            continue

        start = end - WINDOW_LEN + 1

        valid_fraction = (
            present[start:end + 1].mean()
        )

        if valid_fraction < MIN_VALID_FRAC:
            continue

        X.append(
            values[start:end + 1]
        )

        y.append(labels[end])
        ends.append(end_dates[end])

    if not X:
        return None

    return (
        np.stack(X),
        np.asarray(y, dtype=np.int8),
        ends,
    )


# ---------------------------------------------------------------
# Process a batch of drives
# ---------------------------------------------------------------

def process_batch(
    drive_batch,
    batch_number,
    split_name,
):

    keys = list(
        zip(
            drive_batch["model"].to_list(),
            drive_batch["disk_id"].to_list(),
        )
    )

    # Build a small OR filter.
    conditions = [
        (
            (pl.col("model") == model)
            & (pl.col("disk_id") == disk_id)
        )
        for model, disk_id in keys
    ]

    condition = conditions[0]

    for c in conditions[1:]:
        condition = condition | c

    data = (
        pl.scan_parquet(LABELLED)
        .filter(condition)
        .select([
            "model",
            "disk_id",
            "ds",
            "label",
            *FEATURES,
        ])
        .collect()
    )

    X_parts = []
    y_parts = []
    end_parts = []

    for _, g in data.group_by(
        ["model", "disk_id"],
        maintain_order=True,
    ):

        result = make_drive_windows(g)

        if result is None:
            continue

        X, y, ends = result

        X_parts.append(X)
        y_parts.append(y)
        end_parts.extend(ends)

    if not X_parts:
        return 0, 0

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    output_file = (
        OUTPUT_DIR
        / split_name
        / f"batch_{batch_number:05d}.npz"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        end_ds=np.asarray(
            end_parts,
            dtype="U26",
        ),
    )

    return len(y), int(y.sum())


# ---------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------

def process_split(split_name):

    print()
    print("=" * 60)
    print(f"PROCESSING: {split_name}")
    print("=" * 60)

    split_file = (
        SPLITS_DIR
        / f"{split_name}.parquet"
    )

    drives = pl.read_parquet(split_file)

    total_drives = drives.height

    print(
        f"Drives: {total_drives:,}"
    )

    total_windows = 0
    total_positive = 0

    batch_number = 0

    for start in range(
        0,
        total_drives,
        DRIVES_PER_BATCH,
    ):

        end = min(
            start + DRIVES_PER_BATCH,
            total_drives,
        )

        drive_batch = drives[start:end]

        windows, positives = process_batch(
            drive_batch,
            batch_number,
            split_name,
        )

        total_windows += windows
        total_positive += positives

        print(
            f"Batch {batch_number:05d}: "
            f"drives {start + 1:,}-{end:,} | "
            f"windows {windows:,} | "
            f"positives {positives:,}"
        )

        batch_number += 1

    print()
    print(
        f"{split_name} COMPLETE"
    )
    print(
        f"Total windows: {total_windows:,}"
    )
    print(
        f"Total positives: {total_positive:,}"
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":

    # TEST ONLY:
    # We start with STANDARD.
    process_split("standard")