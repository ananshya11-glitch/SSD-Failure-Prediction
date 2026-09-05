from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "dev" / "dev_drives.parquet"
OUTPUT_DIR = ROOT / "data" / "processed" / "dev" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

print("Loading development drives...")
df = pl.read_parquet(INPUT)

# Split healthy and failed drives separately so each split
# keeps the same class balance.
healthy = (
    df.filter(pl.col("drive_label") == 0)
    .sample(fraction=1.0, seed=SEED)
)

failed = (
    df.filter(pl.col("drive_label") == 1)
    .sample(fraction=1.0, seed=SEED)
)


def split_group(group):
    n = group.height

    n_test = int(round(n * 0.20))
    n_cal = int(round(n * 0.16))
    n_train = n - n_test - n_cal

    train = group[:n_train]
    cal = group[n_train:n_train + n_cal]
    test = group[n_train + n_cal:]

    return train, cal, test


healthy_train, healthy_cal, healthy_test = split_group(healthy)
failed_train, failed_cal, failed_test = split_group(failed)

train = pl.concat([healthy_train, failed_train]).sample(
    fraction=1.0,
    seed=SEED,
)

cal = pl.concat([healthy_cal, failed_cal]).sample(
    fraction=1.0,
    seed=SEED,
)

test = pl.concat([healthy_test, failed_test]).sample(
    fraction=1.0,
    seed=SEED,
)

# Save
train.write_parquet(OUTPUT_DIR / "train.parquet")
cal.write_parquet(OUTPUT_DIR / "calibration.parquet")
test.write_parquet(OUTPUT_DIR / "test.parquet")

print()
print("DEV SPLITS CREATED")
print("------------------")

for name, data in [
    ("TRAIN", train),
    ("CALIBRATION", cal),
    ("TEST", test),
]:
    healthy_count = (data["drive_label"] == 0).sum()
    failed_count = (data["drive_label"] == 1).sum()

    print(
        f"{name:12} : "
        f"{data.height} drives "
        f"({healthy_count} healthy, {failed_count} failed)"
    )

# Verify no drive overlap
def keys(data):
    return set(
        zip(
            data["model"].to_list(),
            data["disk_id"].to_list(),
        )
    )

train_keys = keys(train)
cal_keys = keys(cal)
test_keys = keys(test)

print()
print("Overlap checks:")
print("train ∩ calibration:", len(train_keys & cal_keys))
print("train ∩ test       :", len(train_keys & test_keys))
print("calibration ∩ test :", len(cal_keys & test_keys))