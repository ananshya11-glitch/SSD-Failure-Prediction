from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "standard" / "standard_drives.parquet"
OUTPUT_DIR = ROOT / "data" / "processed" / "standard" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def split_group(group):
    n = group.height

    n_test = int(round(n * 0.20))
    n_cal = int(round(n * 0.16))
    n_train = n - n_test - n_cal

    train = group[:n_train]
    calibration = group[n_train:n_train + n_cal]
    test = group[n_train + n_cal:]

    return train, calibration, test


print("Loading standard subset...")

df = pl.read_parquet(INPUT)

healthy = (
    df.filter(pl.col("drive_label") == 0)
    .sample(fraction=1.0, seed=SEED)
)

failed = (
    df.filter(pl.col("drive_label") == 1)
    .sample(fraction=1.0, seed=SEED)
)

healthy_train, healthy_cal, healthy_test = split_group(healthy)
failed_train, failed_cal, failed_test = split_group(failed)

train = pl.concat([healthy_train, failed_train]).sample(
    fraction=1.0,
    seed=SEED,
)

calibration = pl.concat([healthy_cal, failed_cal]).sample(
    fraction=1.0,
    seed=SEED,
)

test = pl.concat([healthy_test, failed_test]).sample(
    fraction=1.0,
    seed=SEED,
)

train.write_parquet(OUTPUT_DIR / "train.parquet")
calibration.write_parquet(OUTPUT_DIR / "calibration.parquet")
test.write_parquet(OUTPUT_DIR / "test.parquet")


def key_set(data):
    return set(
        zip(
            data["model"].to_list(),
            data["disk_id"].to_list(),
        )
    )


train_keys = key_set(train)
cal_keys = key_set(calibration)
test_keys = key_set(test)

print()
print("STANDARD SPLITS CREATED")
print("-----------------------")

for name, data in [
    ("TRAIN", train),
    ("CALIBRATION", calibration),
    ("TEST", test),
]:
    failed_count = int((data["drive_label"] == 1).sum())
    healthy_count = int((data["drive_label"] == 0).sum())

    print(
        f"{name:12}: "
        f"{data.height} drives | "
        f"{healthy_count} healthy | "
        f"{failed_count} failed | "
        f"failure rate={failed_count / data.height:.4%}"
    )

print()
print("Overlap checks:")
print("train ∩ calibration:", len(train_keys & cal_keys))
print("train ∩ test       :", len(train_keys & test_keys))
print("calibration ∩ test :", len(cal_keys & test_keys))