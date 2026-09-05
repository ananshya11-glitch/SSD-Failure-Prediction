from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main():

    FIGURES.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load verified result files
    # ---------------------------------------------------------

    cnn_path = REPORTS / "cnn_results.csv"
    conformal_path = (
        REPORTS / "drive_level_conformal_results.csv"
    )

    if not cnn_path.exists():
        raise FileNotFoundError(
            f"Missing:\n{cnn_path}"
        )

    if not conformal_path.exists():
        raise FileNotFoundError(
            f"Missing:\n{conformal_path}"
        )

    cnn = pd.read_csv(cnn_path)
    conformal = pd.read_csv(conformal_path)

    # ---------------------------------------------------------
    # Validate CNN schema
    # ---------------------------------------------------------

    required_cnn = [
        "Experiment",
        "ROC-AUC",
        "PR-AUC",
        "Precision",
        "Recall",
    ]

    missing = [
        c for c in required_cnn
        if c not in cnn.columns
    ]

    if missing:
        raise ValueError(
            f"cnn_results.csv missing: {missing}"
        )

    # ---------------------------------------------------------
    # Validate conformal schema
    # ---------------------------------------------------------

    required_conformal = [
        "fold",
        "method",
        "coverage",
        "healthy_coverage",
        "failure_coverage",
        "avg_set_size",
    ]

    missing = [
        c for c in required_conformal
        if c not in conformal.columns
    ]

    if missing:
        raise ValueError(
            "drive_level_conformal_results.csv missing: "
            f"{missing}"
        )

    # =========================================================
    # FIGURE 1 — ROC-AUC
    # =========================================================

    experiments = cnn["Experiment"].tolist()
    roc_auc = cnn["ROC-AUC"].tolist()

    plt.figure(figsize=(8, 5))

    plt.bar(
        experiments,
        roc_auc
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1
    )

    plt.ylabel("ROC-AUC")
    plt.xlabel("Evaluation setting")
    plt.title("CNN ROC-AUC Across Evaluation Settings")

    plt.ylim(
        0,
        max(1.0, max(roc_auc) + 0.1)
    )

    plt.tight_layout()

    path = FIGURES / "paper_roc_auc.png"

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================
    # FIGURE 2 — PR-AUC
    # =========================================================

    pr_auc = cnn["PR-AUC"].tolist()

    plt.figure(figsize=(8, 5))

    plt.bar(
        experiments,
        pr_auc
    )

    plt.ylabel("PR-AUC")
    plt.xlabel("Evaluation setting")
    plt.title("CNN PR-AUC Across Evaluation Settings")

    plt.tight_layout()

    path = FIGURES / "paper_pr_auc.png"

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================
    # FIGURE 3 — PRECISION VS RECALL
    # =========================================================

    precision = cnn["Precision"].tolist()
    recall = cnn["Recall"].tolist()

    plt.figure(figsize=(8, 5))

    plt.scatter(
        recall,
        precision,
        s=80
    )

    for i, name in enumerate(experiments):
        plt.annotate(
            name,
            (
                recall[i],
                precision[i]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CNN Precision–Recall Behavior")

    plt.xlim(
        0,
        max(1.0, max(recall) + 0.1)
    )

    plt.ylim(
        0,
        max(0.01, max(precision) + 0.01)
    )

    plt.tight_layout()

    path = FIGURES / "paper_precision_recall.png"

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================
    # FIGURE 4 — FAILURE COVERAGE
    # =========================================================

    conformal_plot = conformal.copy()

    conformal_plot["setting"] = (
        "LOMO-"
        + conformal_plot["fold"].astype(str)
    )

    settings = [
        "LOMO-A",
        "LOMO-B",
        "LOMO-C",
    ]

    methods = [
        "LAC",
        "APS",
    ]

    x = range(len(settings))
    width = 0.35

    lac_values = []
    aps_values = []

    for setting in settings:

        lac = conformal_plot[
            (conformal_plot["setting"] == setting)
            & (conformal_plot["method"] == "LAC")
        ]["failure_coverage"]

        aps = conformal_plot[
            (conformal_plot["setting"] == setting)
            & (conformal_plot["method"] == "APS")
        ]["failure_coverage"]

        if len(lac) != 1:
            raise ValueError(
                f"Expected exactly one LAC result for {setting}."
            )

        if len(aps) != 1:
            raise ValueError(
                f"Expected exactly one APS result for {setting}."
            )

        lac_values.append(
            float(lac.iloc[0])
        )

        aps_values.append(
            float(aps.iloc[0])
        )

    plt.figure(figsize=(8, 5))

    plt.bar(
        [i - width / 2 for i in x],
        lac_values,
        width,
        label="LAC"
    )

    plt.bar(
        [i + width / 2 for i in x],
        aps_values,
        width,
        label="APS"
    )

    plt.axhline(
        0.90,
        linestyle="--",
        linewidth=1,
        label="Nominal 90%"
    )

    plt.xticks(
        list(x),
        settings
    )

    plt.ylabel("Failure coverage")
    plt.xlabel("LOMO setting")
    plt.title(
        "Drive-Level Conformal Failure Coverage "
        "Under Vendor Shift"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FIGURES
        / "paper_conformal_failure_coverage.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================
    # DONE
    # =========================================================

    print()
    print("=" * 80)
    print("PAPER FIGURES CREATED")
    print("=" * 80)

    print()
    print("Figures:")
    print(FIGURES / "paper_roc_auc.png")
    print(FIGURES / "paper_pr_auc.png")
    print(FIGURES / "paper_precision_recall.png")
    print(
        FIGURES
        / "paper_conformal_failure_coverage.png"
    )


if __name__ == "__main__":
    main()