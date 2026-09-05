from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "data" / "processed" / "drive_metadata.parquet"
OUT_DIR = ROOT / "data" / "processed" / "standard"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DRIVES = 20_000
SEED = 42

print("Loading drive metadata...")

df = pl.read_parquet(META)

healthy = df.filter(pl.col("drive_label") == 0)
failed = df.filter(pl.col("drive_label") == 1)

# Preserve the natural drive-level failure prevalence.
failure_rate = failed.height / df.height
n_failed = round(N_DRIVES * failure_rate)
n_healthy = N_DRIVES - n_failed

print(f"Full dataset failure rate: {failure_rate:.4%}")
print(f"Selecting {n_healthy} healthy drives")
print(f"Selecting {n_failed} failed drives")

healthy_sample = healthy.sample(
    n=n_healthy,
    seed=SEED,
)

failed_sample = failed.sample(
    n=n_failed,
    seed=SEED,
)

subset = (
    pl.concat([healthy_sample, failed_sample])
    .sample(fraction=1.0, seed=SEED)
)

output = OUT_DIR / "standard_drives.parquet"

subset.write_parquet(output)

print()
print("STANDARD SUBSET CREATED")
print("------------------------")
print(f"Total drives : {subset.height}")
print(f"Healthy      : {(subset['drive_label'] == 0).sum()}")
print(f"Failed       : {(subset['drive_label'] == 1).sum()}")
print(
    f"Failure rate : "
    f"{(subset['drive_label'] == 1).mean():.4%}"
)
print(f"Output       : {output}")