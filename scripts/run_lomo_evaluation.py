from pathlib import Path
import sys
import json
import gc
import time

import polars as pl

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import Split, validate_split
from src.evaluation.lomo import (
    RunConfig,
    prepare_fold,
    run_cell,
)
from src.conformal import METHODS


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TELEMETRY_ROOT = ROOT / "data" / "processed" / "lomo" / "telemetry"
RESULTS = ROOT / "results" / "lomo_wefr"


# ------------------------------------------------------------
# Exact 16 SMART features used by the project
# ------------------------------------------------------------

FEATURES = [
    "n_5", "r_5",
    "n_9", "r_9",
    "n_12", "r_12",
    "n_183", "r_183",
    "n_184", "r_184",
    "n_187", "r_187",
    "n_197", "r_197",
    "n_199", "r_199",
]


BASE_COLUMNS = [
    "model",
    "disk_id",
    "ds",
    "vendor",
    "label",
] + FEATURES


# ------------------------------------------------------------
# LOMO folds
# ------------------------------------------------------------

FOLDS = {
    "A": "A",
    "B": "B",
    "C": "C",
}


# ------------------------------------------------------------
# Load one telemetry file, keeping ONLY required columns
# ------------------------------------------------------------

def load_telemetry(path: Path) -> pl.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found:\n{path}"
        )

    print(f"Loading: {path}")

    # Inspect schema without loading the whole file.
    schema = pl.read_parquet_schema(path)
    available = set(schema.keys())

    missing = [
        c for c in BASE_COLUMNS
        if c not in available
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}:\n"
            f"{missing}"
        )

    # Only load the columns actually required by the
    # windowing/evaluation pipeline.
    df = pl.read_parquet(
        path,
        columns=BASE_COLUMNS,
    )

    print(
        f"  rows   : {df.height:,}\n"
        f"  drives : "
        f"{df.select(['model', 'disk_id']).unique().height:,}"
    )

    return df


# ------------------------------------------------------------
# Build exact precomputed LOMO split
# ------------------------------------------------------------

def build_fold(vendor: str):

    fold_dir = TELEMETRY_ROOT / f"lomo_{vendor}"

    train_path = fold_dir / "train.parquet"
    cal_path = fold_dir / "calibration.parquet"
    test_path = fold_dir / "test.parquet"

    print()
    print("=" * 70)
    print(f"LOMO-{vendor}")
    print("=" * 70)

    train = load_telemetry(train_path)
    cal = load_telemetry(cal_path)
    test = load_telemetry(test_path)

    # --------------------------------------------------------
    # Combine the three already-defined partitions.
    #
    # IMPORTANT:
    # These are the exact train/cal/test drive partitions
    # produced by the LOMO telemetry stage.
    # --------------------------------------------------------

    labelled = pl.concat(
        [train, cal, test],
        how="vertical",
    )

    # --------------------------------------------------------
    # Build drive-level metadata for Split.
    # --------------------------------------------------------

    def drive_metadata(df):

        return (
            df
            .select(
                [
                    "model",
                    "disk_id",
                    "vendor",
                ]
            )
            .unique(
                subset=["model", "disk_id"],
                keep="first",
            )
        )

    train_meta = drive_metadata(train)
    cal_meta = drive_metadata(cal)
    test_meta = drive_metadata(test)

    split = Split(
        name=f"lomo_{vendor}",
        train=train_meta,
        cal=cal_meta,
        test=test_meta,
        held_out_vendor=vendor,
    )

    # Verify the exact LOMO invariants.
    validate_split(split)

    print()
    print("Split verification:")
    print(
        f"  train drives : {train_meta.height:,}"
    )
    print(
        f"  cal drives   : {cal_meta.height:,}"
    )
    print(
        f"  test drives  : {test_meta.height:,}"
    )
    print(
        f"  held-out     : {vendor}"
    )

    return labelled, split


# ------------------------------------------------------------
# Run one LOMO fold
# ------------------------------------------------------------

def run_fold(vendor: str, cfg: RunConfig):

    labelled, split = build_fold(vendor)

    print()
    print(
        f"Preparing LOMO-{vendor} windows..."
    )

    t0 = time.time()

    fold = prepare_fold(
        labelled,
        split,
        cfg,
    )

    print(
        f"Windows ready in "
        f"{time.time() - t0:.1f}s"
    )

    print(
        f"  train : {len(fold.train):,}"
    )
    print(
        f"  cal   : {len(fold.cal):,}"
    )
    print(
        f"  test  : {len(fold.test):,}"
    )
    print(
        f"  features after WEFR : "
        f"{len(fold.features)}"
    )

    # --------------------------------------------------------
    # Run every conformal method defined by the project.
    # --------------------------------------------------------

    rows = []

    for method in cfg.methods:

        model_name = "random_forest"

        key = cfg.key(
            split.name,
            model_name,
            method,
        )

        output_path = (
            RESULTS / f"{key}.json"
        )

        if output_path.exists():

            print(
                f"\n[cached] "
                f"{split.name} "
                f"{model_name} "
                f"{method}"
            )

            with open(output_path) as f:
                rows.append(json.load(f))

            continue

        print()
        print("-" * 70)
        print(
            f"Running: "
            f"{split.name} | "
            f"{model_name} | "
            f"{method}"
        )
        print("-" * 70)

        t0 = time.time()

        row = run_cell(
            fold,
            split,
            model_name,
            method,
            cfg,
        )

        row["cell_seconds"] = round(
            time.time() - t0,
            2,
        )

        with open(output_path, "w") as f:
            json.dump(
                row,
                f,
                indent=2,
                default=str,
            )

        rows.append(row)

        print(
            f"Coverage : "
            f"{row['empirical_coverage']:.4f}"
        )

        print(
            f"Failure coverage : "
            f"{row['coverage_failure']:.4f}"
        )

        print(
            f"Avg set size : "
            f"{row['avg_set_size']:.3f}"
        )

        print(
            f"Time : "
            f"{row['cell_seconds']:.1f}s"
        )

    # --------------------------------------------------------
    # Release fold memory before next fold.
    # --------------------------------------------------------

    del labelled
    del fold
    del split

    gc.collect()

    return rows


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("LOMO RANDOM FOREST + WEFR EVALUATION")
    print("=" * 70)

    print(
        f"Telemetry : {TELEMETRY_ROOT}"
    )

    print(
        f"Results   : {RESULTS}"
    )

    cfg = RunConfig(
        stride=30,
        flatten_how="last",
        feature_selection=True,
        models=("random_forest",),

        # These are the ACTUAL methods in this repository.
        methods=(
            "split",
            "mondrian_class",
            "mondrian_group",
            "weighted",
        ),

        results_dir=str(RESULTS),
    )

    print()
    print("Configuration:")
    print(f"  stride          : {cfg.stride}")
    print(f"  flatten         : {cfg.flatten_how}")
    print(f"  WEFR            : {cfg.feature_selection}")
    print(f"  model           : {cfg.models}")
    print(f"  score function  : {cfg.score_fn}")
    print(f"  conformal       : {cfg.methods}")
    print()

    # Make sure the requested methods really exist.
    for method in cfg.methods:
        if method not in METHODS:
            raise ValueError(
                f"Unknown conformal method: {method}\n"
                f"Available methods: {tuple(METHODS)}"
            )

    all_rows = []

    for vendor in ("A", "B", "C"):

        rows = run_fold(
            vendor,
            cfg,
        )

        all_rows.extend(rows)

    # --------------------------------------------------------
    # Final combined CSV
    # --------------------------------------------------------

    if all_rows:

        result_df = pl.DataFrame(
            all_rows,
            infer_schema_length=None,
        )

        result_df.write_csv(
            RESULTS / "results.csv"
        )

        print()
        print("=" * 70)
        print("LOMO EVALUATION COMPLETE")
        print("=" * 70)

        print(
            result_df.select(
                [
                    "split",
                    "model",
                    "conformal",
                    "n_test_windows",
                    "empirical_coverage",
                    "coverage_failure",
                    "avg_set_size",
                ]
            )
        )

        print()
        print(
            f"Saved:\n"
            f"{RESULTS / 'results.csv'}"
        )


if __name__ == "__main__":
    main()