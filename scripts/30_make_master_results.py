from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def load_predictions(experiment):
    if experiment == "STANDARD":
        probs_path = RESULTS / "standard_cnn_test_probs.npy"
        labels_path = RESULTS / "standard_cnn_test_labels.npy"
    else:
        fold = experiment[-1]
        probs_path = RESULTS / "lomo" / f"lomo_{fold}_test_probs.npy"
        labels_path = RESULTS / "lomo" / f"lomo_{fold}_test_labels.npy"

    if not probs_path.exists():
        raise FileNotFoundError(f"Missing: {probs_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing: {labels_path}")

    probs = np.load(probs_path)
    labels = np.load(labels_path)

    if len(probs) != len(labels):
        raise ValueError(
            f"{experiment}: probability/label length mismatch"
        )

    return probs, labels


def calculate_cnn_result(experiment):
    probs, labels = load_predictions(experiment)

    predictions = (probs >= 0.5).astype(np.int8)

    return {
        "setting": experiment,
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "n_windows": len(labels),
        "n_positive": int(labels.sum()),
        "positive_rate": labels.mean(),
    }


def main():

    print("=" * 80)
    print("BUILDING MASTER RESULTS TABLE")
    print("=" * 80)

    # =========================================================
    # CNN RESULTS
    # =========================================================

    experiments = [
        "STANDARD",
        "LOMO-A",
        "LOMO-B",
        "LOMO-C",
    ]

    cnn_rows = []

    for experiment in experiments:

        result = calculate_cnn_result(experiment)
        cnn_rows.append(result)

        print()
        print(f"{experiment} CNN verified")
        print(f"  Windows : {result['n_windows']:,}")
        print(f"  Positives: {result['n_positive']:,}")
        print(f"  ROC-AUC : {result['roc_auc']:.4f}")
        print(f"  PR-AUC  : {result['pr_auc']:.4f}")

    cnn = pd.DataFrame(cnn_rows)

    # =========================================================
    # DRIVE-LEVEL CONFORMAL RESULTS
    # =========================================================

    conformal_path = (
        REPORTS / "drive_level_conformal_results.csv"
    )

    if not conformal_path.exists():
        raise FileNotFoundError(
            f"Missing required report:\n{conformal_path}"
        )

    drive_conformal = pd.read_csv(conformal_path)

    expected_conformal = [
        "fold",
        "method",
        "qhat",
        "coverage",
        "healthy_coverage",
        "failure_coverage",
        "avg_set_size",
        "singleton_rate",
        "doubleton_rate",
        "empty_rate",
    ]

    missing = [
        c for c in expected_conformal
        if c not in drive_conformal.columns
    ]

    if missing:
        raise ValueError(
            "drive_level_conformal_results.csv is missing: "
            f"{missing}"
        )

    drive_conformal["setting"] = (
        "LOMO-"
        + drive_conformal["fold"].astype(str)
    )

    drive_conformal = drive_conformal.drop(
        columns=["fold"]
    )

    drive_conformal = drive_conformal.rename(
        columns={
            "coverage": "drive_coverage",
            "healthy_coverage": "drive_healthy_coverage",
            "failure_coverage": "drive_failure_coverage",
            "avg_set_size": "drive_avg_set_size",
        }
    )

    # =========================================================
    # STANDARD ROW
    # =========================================================

    standard = cnn[
        cnn["setting"] == "STANDARD"
    ].copy()

    for column in [
        "method",
        "qhat",
        "drive_coverage",
        "drive_healthy_coverage",
        "drive_failure_coverage",
        "drive_avg_set_size",
        "singleton_rate",
        "doubleton_rate",
        "empty_rate",
    ]:
        standard[column] = pd.NA

    # =========================================================
    # LOMO ROWS
    # =========================================================

    lomo = cnn[
        cnn["setting"].isin(
            ["LOMO-A", "LOMO-B", "LOMO-C"]
        )
    ].copy()

    lomo = lomo.merge(
        drive_conformal,
        on="setting",
        how="left",
        validate="one_to_many",
    )

    if len(lomo) != 6:
        raise ValueError(
            f"Expected 6 LOMO rows after merge, got {len(lomo)}"
        )

    if lomo["method"].isna().any():
        raise ValueError(
            "Conformal merge failed."
        )

    # =========================================================
    # COMBINE
    # =========================================================

    master = pd.concat(
        [
            standard,
            lomo,
        ],
        ignore_index=True,
    )

    # =========================================================
    # INTERPRETATION
    # =========================================================

    def interpretation(setting):

        if setting == "STANDARD":
            return (
                "In-distribution evaluation under natural "
                "class imbalance."
            )

        if setting == "LOMO-A":
            return (
                "Unseen-vendor evaluation shows substantial "
                "distribution-shift effects; LAC coverage "
                "falls below the 90% target."
            )

        if setting == "LOMO-B":
            return (
                "Unseen-vendor evaluation shows near-random "
                "CNN discrimination; LAC maintains high "
                "overall coverage but low failure coverage."
            )

        if setting == "LOMO-C":
            return (
                "Unseen-vendor evaluation shows near-random "
                "CNN discrimination; LAC achieves high "
                "coverage but produces mostly ambiguous "
                "prediction sets."
            )

        return ""

    master["interpretation"] = (
        master["setting"]
        .apply(interpretation)
    )

    # =========================================================
    # COLUMN ORDER
    # =========================================================

    columns = [
        "setting",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "n_windows",
        "n_positive",
        "positive_rate",
        "method",
        "qhat",
        "drive_coverage",
        "drive_healthy_coverage",
        "drive_failure_coverage",
        "drive_avg_set_size",
        "singleton_rate",
        "doubleton_rate",
        "empty_rate",
        "interpretation",
    ]

    master = master[columns]

    # =========================================================
    # SORT
    # =========================================================

    setting_order = {
        "STANDARD": 0,
        "LOMO-A": 1,
        "LOMO-B": 2,
        "LOMO-C": 3,
    }

    method_order = {
        "LAC": 0,
        "APS": 1,
    }

    master["_setting_order"] = (
        master["setting"].map(setting_order)
    )

    master["_method_order"] = (
        master["method"]
        .map(method_order)
        .fillna(-1)
    )

    master = (
        master
        .sort_values(
            [
                "_setting_order",
                "_method_order",
            ]
        )
        .drop(
            columns=[
                "_setting_order",
                "_method_order",
            ]
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # SAVE
    # =========================================================

    output_path = (
        REPORTS / "master_results.csv"
    )

    master.to_csv(
        output_path,
        index=False,
    )

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    print()
    print("=" * 80)
    print("MASTER RESULTS CREATED SUCCESSFULLY")
    print("=" * 80)

    print()
    print(f"Rows in final table: {len(master)}")

    print()
    print(
        master.to_string(
            index=False
        )
    )

    print()
    print(f"Saved:\n{output_path}")


if __name__ == "__main__":
    main()