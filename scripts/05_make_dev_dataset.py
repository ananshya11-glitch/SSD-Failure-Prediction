from pathlib import Path
import polars as pl

# Paths
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "drive_metadata.parquet"
OUT_DIR = ROOT / "data" / "processed" / "dev"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
N_HEALTHY = 5000
N_FAILED = 5000
SEED = 42

print("Loading drive metadata...")
df = pl.read_parquet(META)

# Separate healthy and failed drives
healthy = (
    df.filter(pl.col("drive_label") == 0)
    .sample(n=N_HEALTHY, seed=SEED)
)

failed = (
    df.filter(pl.col("drive_label") == 1)
    .sample(n=N_FAILED, seed=SEED)
)

# Combine and shuffle
dev_drives = (
    pl.concat([healthy, failed])
    .sample(fraction=1.0, seed=SEED)
)

# Save
output = OUT_DIR / "dev_drives.parquet"
dev_drives.write_parquet(output)

print()
print("DEV DATASET CREATED")
print("-------------------")
print(f"Total drives  : {dev_drives.height}")
print(f"Healthy       : {(dev_drives['drive_label'] == 0).sum()}")
print(f"Failed        : {(dev_drives['drive_label'] == 1).sum()}")
print(f"Output        : {output}")