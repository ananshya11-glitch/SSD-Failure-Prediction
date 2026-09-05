from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

LABELLED = ROOT / "data" / "processed" / "alibaba_ssd_labelled.parquet"
DEV_DRIVES = ROOT / "data" / "processed" / "dev" / "dev_drives.parquet"
OUTPUT = ROOT / "data" / "processed" / "dev" / "dev_labelled.parquet"

print("Loading development drive list...")

drives = pl.read_parquet(DEV_DRIVES)

print(f"Development drives: {drives.height}")
print("Reading labelled telemetry...")

# Keep only the columns needed for the development experiment.
# Joining against the 10,000-drive table avoids materializing
# the complete 260M-row dataset in memory.
drive_keys = drives.select(["model", "disk_id"])

lf = (
    pl.scan_parquet(LABELLED)
    .join(
        drive_keys.lazy(),
        on=["model", "disk_id"],
        how="inner",
    )
)

print("Writing development telemetry...")

lf.sink_parquet(
    OUTPUT,
    compression="zstd",
)

print()
print("DEV TELEMETRY CREATED")
print("---------------------")

result = pl.read_parquet(OUTPUT)

print(f"Rows          : {result.height}")
print(f"Columns       : {result.width}")
print(f"Output        : {OUTPUT}")

print()
print("Drive labels:")
print(
    result.group_by("failure_event")
    .agg(pl.len().alias("rows"))
    .sort("failure_event")
)