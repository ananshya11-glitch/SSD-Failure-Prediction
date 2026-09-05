import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.windowing import (
    WindowSet,
    Normalizer,
    flatten,
)
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.windowing import (
    WindowSet,
    Normalizer,
    flatten,
)
from src.features.wefr import select_features
from src.models import MODELS
from src.evaluation.metrics import (
    pick_threshold,
    drive_level_metrics,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

REPORTS.mkdir(
    parents=True,
    exist_ok=True,
)


COMMON_COLS = [
    "n_5",
    "r_5",
    "n_9",
    "r_9",
    "n_12",
    "r_12",
    "n_183",
    "r_183",
    "n_184",
    "r_184",
    "n_187",
    "r_187",
    "n_197",
    "r_197",
    "n_199",
    "r_199",
]


# ============================================================
# LOAD WINDOW SET
# ============================================================

def load_window_set(
    X_path,
    y_path,
    metadata_path,
    stride=30,
):

    X = np.load(
        X_path,
        mmap_mode="r",
    )

    y = np.load(
        y_path,
        mmap_mode="r",
    )

    metadata = pd.read_parquet(
        metadata_path,
    )

    if len(X) != len(y):
        raise ValueError(
            f"X/y mismatch: {len(X)} vs {len(y)}"
        )

    if len(metadata) != len(y):
        raise ValueError(
            f"metadata/y mismatch: "
            f"{len(metadata)} vs {len(y)}"
        )

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3D, got {X.ndim}"
        )

    if X.shape[1] != 30:
        raise ValueError(
            f"Expected 30-day windows, got {X.shape[1]}"
        )

    if X.shape[2] != 16:
        raise ValueError(
            f"Expected 16 features, got {X.shape[2]}"
        )

    # --------------------------------------------------------
    # Build unique drive list.
    # --------------------------------------------------------

    drive_pairs = (
        metadata[
            ["model", "disk_id"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    drive_lookup = {
        (
            row.model,
            int(row.disk_id),
        ): i
        for i, row in drive_pairs.iterrows()
    }

    drive_idx = np.array(
        [
            drive_lookup[
                (
                    row.model,
                    int(row.disk_id),
                )
            ]
            for row in metadata.itertuples()
        ],
        dtype=np.int64,
    )

    drives = [
        (
            row.model,
            int(row.disk_id),
        )
        for row in drive_pairs.itertuples()
    ]

    vendors = np.array(
        [
            str(row.model)[1]
            for row in metadata.itertuples()
        ],
        dtype=object,
    )

    end_ds = metadata[
        "window_date"
    ].to_numpy()

    return WindowSet(
        X=np.asarray(X),
        y=np.asarray(y).astype(np.int8),
        drive_idx=drive_idx,
        end_ds=end_ds,
        drives=drives,
        vendors=vendors,
        features=COMMON_COLS.copy(),
        window_len=30,
        stride=stride,
        positive_stride=stride,
    )


# ============================================================
# LOAD EXPERIMENT
# ============================================================

def load_experiment(setting):

    if setting == "STANDARD":

        base = (
            ROOT
            / "data"
            / "processed"
            / "standard"
            / "windows"
        )

        train = load_window_set(
            base / "train" / "X.npy",
            base / "train" / "y.npy",
            base / "train_metadata.parquet",
        )

        cal = load_window_set(
            base / "calibration" / "X.npy",
            base / "calibration" / "y.npy",
            base / "calibration_metadata.parquet",
        )

        test = load_window_set(
            base / "test" / "X.npy",
            base / "test" / "y.npy",
            base / "test_metadata.parquet",
        )

    else:

        fold = setting[-1]

        base = (
            ROOT
            / "data"
            / "processed"
            / "lomo"
            / "windows"
            / f"lomo_{fold}"
        )

        train = load_window_set(
            base / "train" / "X.npy",
            base / "train" / "y.npy",
            base / "train" / "metadata.parquet",
        )

        cal = load_window_set(
            base / "calibration" / "X.npy",
            base / "calibration" / "y.npy",
            base / "calibration" / "metadata.parquet",
        )

        test = load_window_set(
            base / "test" / "X.npy",
            base / "test" / "y.npy",
            base / "test" / "metadata.parquet",
        )

    return train, cal, test


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_one(setting):

    print()
    print("=" * 80)
    print(f"WEFR COMPARISON: {setting}")
    print("=" * 80)

    train, cal, test = load_experiment(
        setting
    )

    print()
    print("Windows")
    print(
        f"  Train: {len(train):,}"
    )
    print(
        f"  Cal  : {len(cal):,}"
    )
    print(
        f"  Test : {len(test):,}"
    )

    print()
    print("Drives")
    print(
        f"  Train: {train.n_drives:,}"
    )
    print(
        f"  Cal  : {cal.n_drives:,}"
    )
    print(
        f"  Test : {test.n_drives:,}"
    )

    # --------------------------------------------------------
    # NORMALISATION
    #
    # TRAINING DATA ONLY.
    # --------------------------------------------------------

    normalizer = Normalizer.fit(
        train
    )

    train = normalizer.transform(
        train
    )

    cal = normalizer.transform(
        cal
    )

    test = normalizer.transform(
        test
    )

    # --------------------------------------------------------
    # WEFR FEATURE SELECTION
    #
    # TRAINING DATA ONLY.
    # --------------------------------------------------------

    X_train, feature_names = flatten(
        train,
        "last",
    )

    print()
    print(
        f"Initial features: "
        f"{X_train.shape[1]}"
    )

    selection = select_features(
        X_train,
        train.y,
        feature_names,
        max_k=None,
    )

    selected_names = selection.selected

    # Map selected flattened features back to the
    # corresponding SMART attributes.

    selected_features = [
        feature
        for feature in train.features
        if any(
            name.startswith(feature)
            for name in selected_names
        )
    ]

    if not selected_features:
        raise RuntimeError(
            f"{setting}: WEFR selected no usable features."
        )

    indices = [
        train.features.index(feature)
        for feature in selected_features
    ]

    train.X = train.X[
        :,
        :,
        indices,
    ]

    cal.X = cal.X[
        :,
        :,
        indices,
    ]

    test.X = test.X[
        :,
        :,
        indices,
    ]

    train.features = selected_features
    cal.features = selected_features
    test.features = selected_features

    print(
        f"WEFR selected: "
        f"{len(selected_features)} / 16 features"
    )

    print(
        "Selected:"
    )

    for feature in selected_features:
        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # FLATTEN LAST DAY
    #
    # This matches the repository's WEFR/RF evaluation path.
    # --------------------------------------------------------

    X_train, _ = flatten(
        train,
        "last",
    )

    X_cal, _ = flatten(
        cal,
        "last",
    )

    X_test, _ = flatten(
        test,
        "last",
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    #
    # RF is the classifier used as the WEFR backend.
    # --------------------------------------------------------

    model = MODELS[
        "random_forest"
    ](
        seed=42
    ).fit(
        X_train,
        train.y,
    )

    p_train = model.predict_proba(
        X_train
    )

    p_test = model.predict_proba(
        X_test
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Select threshold on TRAINING data only.
    #
    # Never use calibration or test labels to choose
    # the operating point.
    # --------------------------------------------------------

    threshold = pick_threshold(
        p_train,
        train.y,
        beta=0.5,
    )

    print()
    print(
        f"Selected F0.5 threshold: "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # DRIVE-LEVEL METRICS
    #
    # A drive is positive if ANY of its windows exceeds
    # the selected threshold.
    # --------------------------------------------------------

    metrics = drive_level_metrics(
        p_test,
        test.y,
        test.drive_idx,
        threshold,
        beta=0.5,
    )

    print()
    print("=" * 80)
    print(f"{setting} DRIVE-LEVEL RESULTS")
    print("=" * 80)

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F0.5      : "
        f"{metrics['f0.5']:.4f}"
    )

    print(
        f"Threshold : "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    return {
        "setting": setting,
        "evaluation_unit": "drive",
        "model": "random_forest",
        "feature_selection": "WEFR",
        "n_features": len(selected_features),
        "selected_features": ",".join(
            selected_features
        ),
        "threshold": threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f0.5": metrics["f0.5"],
        "n_test_drives": test.n_drives,
        "n_test_failed_drives": int(
            test.y[
                test.drive_idx
                == test.drive_idx
            ].sum()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("WEFR / RANDOM FOREST DRIVE-LEVEL COMPARISON")
    print("=" * 80)

    settings = [
        "STANDARD",
        "LOMO-A",
        "LOMO-B",
        "LOMO-C",
    ]

    rows = []

    for setting in settings:

        rows.append(
            run_one(setting)
        )

    results = pd.DataFrame(
        rows
    )

    output = (
        REPORTS
        / "wefr_drive_level_results.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL WEFR COMPARISON")
    print("=" * 80)

    print(
        results[
            [
                "setting",
                "evaluation_unit",
                "model",
                "feature_selection",
                "n_features",
                "threshold",
                "precision",
                "recall",
                "f0.5",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved:\n{output}"
    )


if __name__ == "__main__":
    main()