from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "alibaba_ssd_labelled.parquet"
SPLIT_DIR = ROOT / "data" / "processed" / "standard" / "splits"
OUTPUT_DIR = ROOT / "data" / "processed" / "standard" / "telemetry"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000


def extract_split(name):
    print()
    print("=" * 60)
    print(f"EXTRACTING {name.upper()}")
    print("=" * 60)

    split_file = SPLIT_DIR / f"{name}.parquet"
    output_file = OUTPUT_DIR / f"{name}.parquet"

    # --------------------------------------------------------
    # Load only the drive IDs for this split.
    # --------------------------------------------------------

    drives = pl.read_parquet(split_file)

    drive_keys = set(
        zip(
            drives["model"].to_list(),
            drives["disk_id"].to_list(),
        )
    )

    print(f"Drives to extract: {len(drive_keys):,}")
    print(f"Reading: {INPUT}")

    # --------------------------------------------------------
    # Read the huge Parquet file in batches.
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(INPUT)

    writer = None
    rows_written = 0
    batches = 0

    try:

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            batches += 1

            table = pa.Table.from_batches([batch])

            # Convert only this batch to Polars.
            data = pl.from_arrow(table)

            # Create a temporary key for matching drives.
            keys = list(
                zip(
                    data["model"].to_list(),
                    data["disk_id"].to_list(),
                )
            )

            mask = [
                key in drive_keys
                for key in keys
            ]

            if any(mask):

                filtered = data.filter(
                    pl.Series(
                        "keep",
                        mask,
                    )
                )

                if filtered.height > 0:

                    arrow_table = filtered.to_arrow()

                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_file,
                            arrow_table.schema,
                            compression="zstd",
                        )

                    writer.write_table(
                        arrow_table
                    )

                    rows_written += filtered.height

            if batches % 100 == 0:

                print(
                    f"Processed batches: {batches:,} | "
                    f"Rows written: {rows_written:,}"
                )

    finally:

        if writer is not None:
            writer.close()

    print()
    print(f"{name.upper()} COMPLETE")
    print(f"Rows written: {rows_written:,}")
    print(f"Output: {output_file}")


for split in [
    "train",
    "calibration",
    "test",
]:
    extract_split(split)


print()
print("=" * 60)
print("STANDARD TELEMETRY EXTRACTION COMPLETE")
print("=" * 60)