

"""
Verification and profiling of alibaba_ssd_processed.parquet

Run this BEFORE building labels or splits. It checks for the failure
modes that corrupt downstream results silently, and produces the
per-vendor profiling table that decides whether LOMO is viable.

Outputs:
  reports/column_order_check.txt
  reports/vendor_profile.csv
  reports/attribute_availability.csv
"""

import glob
import os

import polars as pl

RAW_DIRS = ["data/raw/smartlog2018ssd", "data/raw/smartlog2019ssd"]
PARQUET = "data/processed/alibaba_ssd_processed.parquet"
REPORTS = "reports"

os.makedirs(REPORTS, exist_ok=True)


# ---------------------------------------------------------------
# CHECK 1: column order consistency across raw daily CSVs
#
# pq writer cast(..., safe=False) maps fields BY POSITION.
# If any file has a different column order, its values were
# written under the wrong names. This check is non-negotiable.
# ---------------------------------------------------------------

def check_column_order():
    files = []
    for d in RAW_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))

    if not files:
        raise FileNotFoundError("No raw CSVs found for order check.")

    reference = pl.read_csv(files[0], n_rows=0).columns
    mismatches = []

    for f in files:
        cols = pl.read_csv(f, n_rows=0).columns
        if cols != reference:
            same_set = set(cols) == set(reference)
            mismatches.append(
                (os.path.basename(f),
                 "REORDERED" if same_set else "DIFFERENT COLUMN SET")
            )

    with open(os.path.join(REPORTS, "column_order_check.txt"), "w") as fh:
        fh.write(f"Files checked: {len(files)}\n")
        fh.write(f"Reference: {os.path.basename(files[0])}\n")
        fh.write(f"Column count: {len(reference)}\n\n")
        if mismatches:
            fh.write("!!! MISMATCHES FOUND — parquet may be corrupted !!!\n")
            for name, kind in mismatches:
                fh.write(f"  {name}: {kind}\n")
        else:
            fh.write("OK: all files share identical column order.\n")

    print(f"[1] Column order: {len(mismatches)} mismatch(es)")
    if mismatches:
        print("    ACTION REQUIRED: re-run ingest with name-based alignment.")
    return len(mismatches) == 0


# ---------------------------------------------------------------
# CHECK 2: duplicate failure records (causes row fan-out on merge)
# ---------------------------------------------------------------

def check_failure_duplicates(failure_csv):
    fail = pl.read_csv(failure_csv)
    n = fail.height
    n_unique = fail.select(["model", "disk_id"]).unique().height
    print(f"[2] Failure rows: {n:,} | unique (model, disk_id): {n_unique:,}")
    if n != n_unique:
        print(f"    WARNING: {n - n_unique} duplicate keys -> merge fan-out.")
    return n == n_unique


# ---------------------------------------------------------------
# CHECK 3: is disk_id unique on its own, or only within model?
# ---------------------------------------------------------------

def check_drive_key(lf):
    pairs = (
        lf.select(["disk_id", "model"])
        .unique()
        .collect(streaming=True)
    )
    collisions = (
        pairs.group_by("disk_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    print(f"[3] disk_ids appearing under >1 model: {collisions:,}")
    if collisions:
        print("    Confirmed: drive key MUST be (model, disk_id).")
    return collisions


# ---------------------------------------------------------------
# CHECK 4: post-failure telemetry and censoring
# ---------------------------------------------------------------

def check_temporal_integrity(lf):
    post = (
        lf.filter(pl.col("days_to_failure") < 0)
        .select(pl.len())
        .collect(streaming=True)
        .item()
    )
    print(f"[4] Rows after recorded failure: {post:,}  (decide: drop?)")

    last_seen = (
        lf.group_by(["model", "disk_id"])
        .agg(pl.col("ds").max().alias("last_ds"))
        .collect(streaming=True)
    )
    global_end = last_seen["last_ds"].max()
    early_exit = last_seen.filter(pl.col("last_ds") < global_end).height
    print(f"    Global last date: {global_end}")
    print(f"    Drives whose telemetry stops early: {early_exit:,}")
    print("    -> these are right-censored, not confirmed healthy.")
    return last_seen


# ---------------------------------------------------------------
# PROFILE: the table that decides whether LOMO is viable
# ---------------------------------------------------------------

def profile_vendors(lf):
    per_model = (
        lf.with_columns(
            pl.col("model").str.slice(1, 1).alias("vendor")
        )
        .group_by(["vendor", "model"])
        .agg(
            pl.col("disk_id").n_unique().alias("n_drives"),
            pl.len().alias("n_rows"),
            pl.col("ds").min().alias("first_ds"),
            pl.col("ds").max().alias("last_ds"),
            pl.col("failure_time").is_not_null().sum().alias("failed_rows"),
        )
        .collect(streaming=True)
    )

    failed_drives = (
        lf.filter(pl.col("failure_time").is_not_null())
        .with_columns(pl.col("model").str.slice(1, 1).alias("vendor"))
        .group_by(["vendor", "model"])
        .agg(pl.col("disk_id").n_unique().alias("n_failed_drives"))
        .collect(streaming=True)
    )

    out = (
        per_model.join(failed_drives, on=["vendor", "model"], how="left")
        .with_columns(pl.col("n_failed_drives").fill_null(0))
        .with_columns(
            (pl.col("n_failed_drives") / pl.col("n_drives") * 100)
            .round(3)
            .alias("failure_pct")
        )
        .sort(["vendor", "model"])
    )

    out.write_csv(os.path.join(REPORTS, "vendor_profile.csv"))
    print("\n[PROFILE] per model:")
    print(out)

    by_vendor = (
        out.group_by("vendor")
        .agg(
            pl.col("n_drives").sum(),
            pl.col("n_failed_drives").sum(),
        )
        .sort("vendor")
    )
    print("\n[PROFILE] per vendor (this decides LOMO viability):")
    print(by_vendor)
    return out


# ---------------------------------------------------------------
# PROFILE: which SMART attributes each vendor actually populates
# ---------------------------------------------------------------

def attribute_availability(lf):
    meta = {"disk_id", "ds", "model", "failure_time",
            "days_to_failure", "failure_event"}
    smart_cols = [c for c in lf.collect_schema().names() if c not in meta]

    result = (
        lf.with_columns(pl.col("model").str.slice(1, 1).alias("vendor"))
        .group_by("vendor")
        .agg([
            pl.col(c).is_not_null().mean().round(4).alias(c)
            for c in smart_cols
        ])
        .collect(streaming=True)
        .sort("vendor")
    )

    result.write_csv(os.path.join(REPORTS, "attribute_availability.csv"))
    print(f"\n[PROFILE] attribute availability written "
          f"({len(smart_cols)} SMART columns)")
    return result


# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    order_ok = check_column_order()

    fail_csv = glob.glob(
        "data/raw/ssd_failure_label.csv/**/*.csv", recursive=True
    )
    if fail_csv:
        check_failure_duplicates(fail_csv[0])

    lf = pl.scan_parquet(PARQUET)

    check_drive_key(lf)
    check_temporal_integrity(lf)

    print("\n" + "=" * 60)
    print("PROFILING")
    print("=" * 60)

    profile_vendors(lf)
    attribute_availability(lf)

    print("\nDone. Reports in ./reports/")
    if not order_ok:
        print("\n*** Do not proceed until column order is resolved. ***")
