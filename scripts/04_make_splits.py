from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "drive_metadata.parquet"
OUTPUT = ROOT / "data" / "processed" / "splits"

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRAC = 0.20
CAL_FRAC = 0.20

VENDORS = ["A", "B", "C"]


# ---------------------------------------------------------------
# Stratified assignment
# ---------------------------------------------------------------

def stratified_assign(
    drives: pl.DataFrame,
    fractions: dict[str, float],
    seed: int,
) -> pl.DataFrame:

    rng = np.random.default_rng(seed)
    names = list(fractions)
    pieces = []

    for _, group in drives.group_by(
        ["model", "drive_label"],
        maintain_order=True,
    ):
        idx = rng.permutation(group.height)
        group = group[idx]

        bounds = []
        accumulated = 0.0

        for name in names[:-1]:
            accumulated += fractions[name]
            bounds.append(
                int(round(accumulated * group.height))
            )

        starts = [0] + bounds
        ends = bounds + [group.height]

        for name, start, end in zip(
            names,
            starts,
            ends,
        ):
            if end > start:
                pieces.append(
                    group[start:end].with_columns(
                        pl.lit(name).alias("split")
                    )
                )

    return pl.concat(pieces)


# ---------------------------------------------------------------
# STANDARD split
# ---------------------------------------------------------------

def make_standard(drives: pl.DataFrame) -> pl.DataFrame:

    fractions = {
        "test": TEST_FRAC,
        "cal": (1 - TEST_FRAC) * CAL_FRAC,
        "train": (1 - TEST_FRAC) * (1 - CAL_FRAC),
    }

    return stratified_assign(
        drives,
        fractions,
        SEED,
    )


# ---------------------------------------------------------------
# LOMO split
# ---------------------------------------------------------------

def make_lomo(
    drives: pl.DataFrame,
    held_out_vendor: str,
) -> pl.DataFrame:

    test = drives.filter(
        pl.col("vendor") == held_out_vendor
    ).with_columns(
        pl.lit("test").alias("split")
    )

    pool = drives.filter(
        pl.col("vendor") != held_out_vendor
    )

    train_cal = stratified_assign(
        pool,
        {
            "cal": CAL_FRAC,
            "train": 1 - CAL_FRAC,
        },
        SEED,
    )

    return pl.concat(
        [train_cal, test],
        how="vertical",
    )


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

def validate_split(
    split: pl.DataFrame,
    name: str,
    held_out_vendor: str | None = None,
) -> None:

    # Every drive must occur exactly once.
    counts = (
        split
        .group_by(["model", "disk_id"])
        .agg(pl.len().alias("n"))
    )

    assert counts["n"].min() == 1
    assert counts["n"].max() == 1

    # Every expected split must exist.
    parts = set(split["split"].unique())

    assert "train" in parts
    assert "cal" in parts
    assert "test" in parts

    # Train / calibration / test must be drive-disjoint.
    keys = {}

    for part in ["train", "cal", "test"]:
        keys[part] = set(
            map(
                tuple,
                split
                .filter(pl.col("split") == part)
                .select(["model", "disk_id"])
                .iter_rows(),
            )
        )

    assert not keys["train"] & keys["cal"]
    assert not keys["train"] & keys["test"]
    assert not keys["cal"] & keys["test"]

    # LOMO-specific validation.
    if held_out_vendor is not None:

        train_vendors = set(
            split
            .filter(pl.col("split") == "train")["vendor"]
            .unique()
        )

        cal_vendors = set(
            split
            .filter(pl.col("split") == "cal")["vendor"]
            .unique()
        )

        test_vendors = set(
            split
            .filter(pl.col("split") == "test")["vendor"]
            .unique()
        )

        assert held_out_vendor not in train_vendors
        assert held_out_vendor not in cal_vendors
        assert test_vendors == {held_out_vendor}

    print(f"\n{name}")
    print("-" * len(name))

    print(
        split
        .group_by("split")
        .agg([
            pl.len().alias("drives"),
            pl.col("drive_label").sum().alias("failed_drives"),
        ])
        .sort("split")
    )

    print(
        split
        .group_by(["split", "vendor"])
        .agg([
            pl.len().alias("drives"),
            pl.col("drive_label").sum().alias("failed_drives"),
        ])
        .sort(["split", "vendor"])
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():

    print("Loading drive metadata...")
    drives = pl.read_parquet(INPUT)

    print(f"Drives loaded: {drives.height:,}")

    # STANDARD
    standard = make_standard(drives)

    validate_split(
        standard,
        "STANDARD",
    )

    standard.write_parquet(
        OUTPUT / "standard.parquet"
    )

    # LOMO
    for vendor in VENDORS:

        lomo = make_lomo(
            drives,
            vendor,
        )

        validate_split(
            lomo,
            f"LOMO-{vendor}",
            held_out_vendor=vendor,
        )

        lomo.write_parquet(
            OUTPUT / f"lomo_{vendor}.parquet"
        )

    print("\nAll splits saved.")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()