from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "standard"
    / "telemetry"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "lomo"
    / "telemetry"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


VENDORS = ["A", "B", "C"]

SPLITS = [
    "train",
    "calibration",
    "test",
]


def make_lomo(excluded_vendor):

    print()
    print("=" * 60)
    print(
        f"CREATING LOMO-{excluded_vendor}"
    )
    print("=" * 60)

    for split in SPLITS:

        input_file = (
            INPUT_DIR
            / f"{split}.parquet"
        )

        output_dir = (
            OUTPUT_DIR
            / f"lomo_{excluded_vendor}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir
            / f"{split}.parquet"
        )

        df = pl.read_parquet(
            input_file
        )

        # ----------------------------------------------------
        # LOMO rule:
        #
        # train/calibration:
        #     exclude held-out vendor
        #
        # test:
        #     ONLY held-out vendor
        # ----------------------------------------------------

        if split in ["train", "calibration"]:

            df = df.filter(
                pl.col("vendor")
                != excluded_vendor
            )

        else:

            df = df.filter(
                pl.col("vendor")
                == excluded_vendor
            )

        drives = (
            df
            .select(
                ["model", "disk_id"]
            )
            .unique()
            .height
        )

        positives = int(
            df["label"].sum()
        )

        print()
        print(
            f"LOMO-{excluded_vendor} "
            f"{split.upper()}"
        )

        print(
            f"Rows: {df.height:,}"
        )

        print(
            f"Drives: {drives:,}"
        )

        print(
            f"Positive rows: "
            f"{positives:,}"
        )

        df.write_parquet(
            output_file,
            compression="zstd"
        )

        print(
            f"Saved: {output_file}"
        )


for vendor in VENDORS:
    make_lomo(vendor)


print()
print("=" * 60)
print("LOMO TELEMETRY CREATION COMPLETE")
print("=" * 60)