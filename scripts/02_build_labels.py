import os
import polars as pl
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/alibaba_ssd_processed.parquet"
OUTPUT_FILE = "data/processed/alibaba_ssd_labelled.parquet"

HORIZON = 30
MIN_HISTORY = 30
BATCH_SIZE = 100_000

DRIVE_KEY = ["model", "disk_id"]


print("=" * 60)
print("PHASE 2 LABEL CONSTRUCTION")
print("=" * 60)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Horizon: {HORIZON} days")
print(f"Minimum history: {MIN_HISTORY} days")


# ============================================================
# PASS 1
# BUILD SMALL DRIVE-LEVEL METADATA TABLE
# ============================================================

print("\nPASS 1: Building drive-level metadata...")
print("Only model, disk_id, ds and failure_time are read.")

lf = pl.scan_parquet(INPUT_FILE)

drive_info = (
    lf
    .select([
        "model",
        "disk_id",
        "ds",
        "failure_time",
    ])
    .group_by(DRIVE_KEY)
    .agg([
        pl.col("ds").max().alias("_last_ds"),
        pl.col("failure_time").max().alias("_drive_failure_time"),
    ])
    .collect(engine="streaming")
)

print(f"Drives found: {drive_info.height:,}")


# ============================================================
# PASS 1B
# FIND DRIVES WITH ENOUGH USABLE HISTORY
# ============================================================

print("\nPASS 1b: Counting usable rows per drive...")


usable_counts = (
    lf
    .select([
        "model",
        "disk_id",
        "ds",
        "failure_time",
    ])

    # drive_info is a normal DataFrame, so convert it
    # back to LazyFrame before joining.
    .join(
        drive_info.lazy(),
        on=DRIVE_KEY,
        how="left",
    )

    # --------------------------------------------------------
    # Calendar-day days-to-failure
    # --------------------------------------------------------

    .with_columns(
        (
            pl.col("failure_time").dt.date()
            - pl.col("ds").dt.date()
        )
        .dt.total_days()
        .cast(pl.Float64)
        .alias("_dtf")
    )

    # --------------------------------------------------------
    # Keep only rows whose labels are observable.
    #
    # Failed drives:
    #     strictly before failure.
    #
    # Non-failed drives:
    #     at least 30 days before their final observation.
    # --------------------------------------------------------

    .filter(
        (
            pl.col("failure_time").is_not_null()
            & (pl.col("_dtf") > 0)
        )
        |
        (
            pl.col("failure_time").is_null()
            & (
                (
                    pl.col("_last_ds") - pl.col("ds")
                )
                .dt.total_days()
                >= HORIZON
            )
        )
    )

    # --------------------------------------------------------
    # Count usable rows per drive
    # --------------------------------------------------------

    .group_by(DRIVE_KEY)
    .agg(
        pl.len().alias("_usable_rows")
    )

    .filter(
        pl.col("_usable_rows") >= MIN_HISTORY
    )

    .collect(engine="streaming")
)


print(
    f"Drives with >= {MIN_HISTORY} usable rows: "
    f"{usable_counts.height:,}"
)


# Only the drive key is needed for the second pass.
eligible_drives = usable_counts.select(DRIVE_KEY)

print(
    f"Eligible drive table size: "
    f"{eligible_drives.height:,}"
)


# ============================================================
# REMOVE OLD OUTPUT IF PRESENT
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


# ============================================================
# PASS 2
# READ AND WRITE THE LARGE DATASET IN BATCHES
# ============================================================

print("\nPASS 2: Writing labelled Parquet in batches...")
print(f"Batch size: {BATCH_SIZE:,}")
print()


parquet_file = pq.ParquetFile(INPUT_FILE)

writer = None

total_in = 0
total_out = 0
total_positive = 0
batch_number = 0


# ============================================================
# PROCESS EACH ARROW BATCH
# ============================================================

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE
):

    batch_number += 1

    # Convert ONLY this batch to Polars.
    df = pl.from_arrow(batch)

    total_in += df.height


    # --------------------------------------------------------
    # ADD VENDOR
    #
    # MA1 / MA2 -> A
    # MB1 / MB2 -> B
    # MC1 / MC2 -> C
    # --------------------------------------------------------

    df = df.with_columns(
        pl.col("model")
        .str.slice(1, 1)
        .alias("vendor")
    )


    # --------------------------------------------------------
    # CALENDAR-DAY DAYS-TO-FAILURE
    #
    # SMART data is daily, so use calendar dates rather
    # than fractional timestamp differences.
    # --------------------------------------------------------

    df = df.with_columns(
        (
            pl.col("failure_time").dt.date()
            - pl.col("ds").dt.date()
        )
        .dt.total_days()
        .cast(pl.Float64)
        .alias("days_to_failure")
    )


    # --------------------------------------------------------
    # KEEP ONLY DRIVES THAT HAVE >= 30 USABLE ROWS
    # --------------------------------------------------------

    df = df.join(
        eligible_drives,
        on=DRIVE_KEY,
        how="inner",
    )


    # --------------------------------------------------------
    # REMOVE UNOBSERVABLE TAIL FOR NON-FAILED DRIVES
    #
    # This must be calculated using the drive's own last date.
    #
    # We obtain the last date from the small drive_info table.
    # --------------------------------------------------------

    df = df.join(
        drive_info.select(
            DRIVE_KEY + ["_last_ds"]
        ),
        on=DRIVE_KEY,
        how="left",
    )

    df = df.filter(
        pl.col("failure_time").is_not_null()
        |
        (
            (
                pl.col("_last_ds") - pl.col("ds")
            )
            .dt.total_days()
            >= HORIZON
        )
    )


    # --------------------------------------------------------
    # REMOVE FAILURE-DAY AND POST-FAILURE ROWS
    #
    # Only observations strictly before failure are predictions.
    # --------------------------------------------------------

    df = df.filter(
        pl.col("failure_time").is_null()
        |
        (pl.col("days_to_failure") > 0)
    )


    # --------------------------------------------------------
    # CREATE LABEL
    #
    # 1 = failure within the next 30 calendar days
    # 0 = otherwise
    # --------------------------------------------------------

    df = df.with_columns(
        (
            (pl.col("days_to_failure") > 0)
            &
            (pl.col("days_to_failure") <= HORIZON)
        )
        .fill_null(False)
        .cast(pl.Int8)
        .alias("label")
    )


    # Remove helper column.
    df = df.drop("_last_ds")


    # --------------------------------------------------------
    # UPDATE COUNTERS
    # --------------------------------------------------------

    total_out += df.height
    total_positive += int(df["label"].sum())


    # --------------------------------------------------------
    # WRITE CURRENT BATCH
    # --------------------------------------------------------

    table = df.to_arrow()

    if writer is None:

        writer = pq.ParquetWriter(
            OUTPUT_FILE,
            table.schema,
            compression="snappy",
        )

    else:

        table = table.cast(
            writer.schema,
            safe=False,
        )

    writer.write_table(table)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if batch_number % 10 == 0:

        print(
            f"batch {batch_number:>5,} | "
            f"input {total_in:>12,} | "
            f"output {total_out:>12,} | "
            f"positives {total_positive:>10,}"
        )


# ============================================================
# CLOSE PARQUET WRITER
# ============================================================

if writer is not None:
    writer.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LABEL CONSTRUCTION COMPLETE")
print("=" * 60)

print(f"Input rows processed : {total_in:,}")
print(f"Output rows          : {total_out:,}")
print(f"Positive rows        : {total_positive:,}")

if total_out:
    print(
        f"Positive rate        : "
        f"{100 * total_positive / total_out:.4f}%"
    )

print(f"Output file          : {OUTPUT_FILE}")