from pathlib import Path
import polars as pl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TELEMETRY = ROOT / "data" / "processed" / "dev" / "dev_labelled.parquet"
SPLIT_DIR = ROOT / "data" / "processed" / "dev" / "splits"
WINDOW_DIR = ROOT / "data" / "processed" / "dev" / "windows"

WINDOW_DIR.mkdir(parents=True, exist_ok=True)

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

WINDOW_LEN = 30


def make_windows(split_name):

    print()
    print("=" * 60)
    print(f"CREATING WINDOWS: {split_name.upper()}")
    print("=" * 60)

    split_file = SPLIT_DIR / f"{split_name}.parquet"

    # Get the drives belonging to this split.
    drives = pl.read_parquet(split_file)

    drive_keys = drives.select(["model", "disk_id"])

    print(f"Drives in split: {drives.height}")

    # Extract only this split's telemetry.
    print("Extracting telemetry...")

    df = (
        pl.scan_parquet(TELEMETRY)
        .join(
            drive_keys.lazy(),
            on=["model", "disk_id"],
            how="inner",
        )
        .collect()
    )

    print(f"Telemetry rows: {df.height}")

    # Process one drive at a time.
    groups = df.partition_by(
        ["model", "disk_id"],
        maintain_order=False,
    )

    print(f"Processing {len(groups)} drives...")

    X_parts = []
    y_parts = []

    total_windows = 0

    for i, drive in enumerate(groups, start=1):

        drive = drive.sort("ds")

        if drive.height < WINDOW_LEN:
            continue

        values = (
            drive
            .select(FEATURES)
            .to_numpy()
            .astype(np.float32)
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        labels = drive["label"].to_numpy().astype(np.int8)

        n_windows = drive.height - WINDOW_LEN + 1

        # Create windows for this drive only.
        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=WINDOW_LEN,
            axis=0,
        )

        # sliding_window_view puts dimensions as:
        # (number_of_windows, number_of_features, window_length)
        # We need:
        # (number_of_windows, window_length, number_of_features)
        windows = np.transpose(windows, (0, 2, 1))

        window_labels = labels[WINDOW_LEN - 1:]

        X_parts.append(windows)
        y_parts.append(window_labels)

        total_windows += n_windows

        if i % 250 == 0:
            print(
                f"Processed {i}/{len(groups)} drives "
                f"({i / len(groups) * 100:.1f}%) | "
                f"windows: {total_windows:,}"
            )

    print()
    print("Combining windows...")

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)

    x_path = WINDOW_DIR / f"{split_name}_X.npy"
    y_path = WINDOW_DIR / f"{split_name}_y.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    print()
    print(f"{split_name.upper()} WINDOWS CREATED")
    print("-----------------------------")
    print(f"X shape           : {X.shape}")
    print(f"y shape           : {y.shape}")
    print(f"Positive windows  : {(y == 1).sum():,}")
    print(f"Negative windows  : {(y == 0).sum():,}")
    print(f"X size            : {X.nbytes / (1024**3):.2f} GB")
    print(f"Saved X           : {x_path}")
    print(f"Saved y           : {y_path}")

    # Release memory before processing the next split.
    del df
    del groups
    del X_parts
    del y_parts
    del X
    del y


for split in ["train", "calibration", "test"]:
    make_windows(split)

print()
print("=" * 60)
print("ALL DEVELOPMENT WINDOWS CREATED")
print("=" * 60)