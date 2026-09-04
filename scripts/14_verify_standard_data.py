from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "processed" / "standard" / "splits"
DATA_DIR = ROOT / "data" / "processed" / "standard" / "telemetry"

splits = ["train", "calibration", "test"]

print("=" * 60)
print("VERIFYING STANDARD TELEMETRY")
print("=" * 60)

all_keys = []

for split in splits:
    split_file = SPLIT_DIR / f"{split}.parquet"
    data_file = DATA_DIR / f"{split}.parquet"

    expected = pl.read_parquet(split_file)
    actual = pl.read_parquet(data_file)

    expected_keys = expected.select(["model", "disk_id"])
    actual_keys = actual.select(["model", "disk_id"]).unique()

    expected_set = set(
        zip(
            expected_keys["model"].to_list(),
            expected_keys["disk_id"].to_list(),
        )
    )

    actual_set = set(
        zip(
            actual_keys["model"].to_list(),
            actual_keys["disk_id"].to_list(),
        )
    )

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    print()
    print(f"{split.upper()}")
    print("-" * 40)
    print(f"Expected drives : {len(expected_set):,}")
    print(f"Actual drives   : {len(actual_set):,}")
    print(f"Missing drives  : {len(missing):,}")
    print(f"Extra drives    : {len(extra):,}")
    print(f"Rows            : {actual.height:,}")

    # Check that every row belongs to a requested drive.
    if extra:
        print("ERROR: Extra drives found!")
        raise SystemExit(1)

    if missing:
        print("ERROR: Missing drives found!")
        raise SystemExit(1)

    # Check required columns.
    required = {
        "model",
        "disk_id",
        "ds",
        "failure_time",
        "days_to_failure",
        "failure_event",
        "vendor",
        "label",
    }

    missing_columns = required - set(actual.columns)

    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        raise SystemExit(1)

    # Check drive labels if present.
    if "drive_label" in actual.columns:
        labels = (
            actual
            .select(["model", "disk_id", "drive_label"])
            .unique()
        )

        inconsistent = (
            labels
            .group_by(["model", "disk_id"])
            .agg(pl.col("drive_label").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )

        if inconsistent.height > 0:
            print("ERROR: Inconsistent drive labels!")
            raise SystemExit(1)

    all_keys.append(actual.select(["model", "disk_id"]))


# ------------------------------------------------------------
# Check overlap between train / calibration / test.
# ------------------------------------------------------------

train_keys = set(
    zip(
        all_keys[0]["model"].to_list(),
        all_keys[0]["disk_id"].to_list(),
    )
)

cal_keys = set(
    zip(
        all_keys[1]["model"].to_list(),
        all_keys[1]["disk_id"].to_list(),
    )
)

test_keys = set(
    zip(
        all_keys[2]["model"].to_list(),
        all_keys[2]["disk_id"].to_list(),
    )
)

train_cal = train_keys & cal_keys
train_test = train_keys & test_keys
cal_test = cal_keys & test_keys

print()
print("=" * 60)
print("OVERLAP CHECK")
print("=" * 60)

print(f"Train ∩ Calibration : {len(train_cal):,}")
print(f"Train ∩ Test        : {len(train_test):,}")
print(f"Calibration ∩ Test  : {len(cal_test):,}")

if train_cal or train_test or cal_test:
    print("ERROR: Drive overlap detected!")
    raise SystemExit(1)


print()
print("=" * 60)
print("ALL STANDARD TELEMETRY CHECKS PASSED")
print("=" * 60)