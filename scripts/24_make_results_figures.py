from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

cnn = pd.read_csv(
    REPORTS / "cnn_results.csv"
)

conformal = pd.read_csv(
    REPORTS / "conformal_results.csv"
)


# ============================================================
# 1. ROC-AUC
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    cnn["Experiment"],
    cnn["ROC-AUC"]
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Random baseline"
)

plt.ylabel("ROC-AUC")
plt.title("ROC-AUC Across Evaluation Settings")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "roc_auc_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. PR-AUC
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    cnn["Experiment"],
    cnn["PR-AUC"]
)

plt.ylabel("PR-AUC")
plt.title("PR-AUC Across Evaluation Settings")

plt.tight_layout()

plt.savefig(
    FIGURES / "pr_auc_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. PRECISION AND RECALL
# ============================================================

x = range(len(cnn))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    cnn["Precision"],
    width,
    label="Precision"
)

plt.bar(
    [i + width / 2 for i in x],
    cnn["Recall"],
    width,
    label="Recall"
)

plt.xticks(
    list(x),
    cnn["Experiment"]
)

plt.ylabel("Score")
plt.title("Precision and Recall Across Evaluation Settings")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "precision_recall_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. CONFORMAL FAILURE COVERAGE — LAC
# ============================================================

lac = conformal[
    conformal["Method"] == "LAC"
]

plt.figure(figsize=(8, 5))

plt.bar(
    lac["Experiment"],
    lac["Failure Coverage"]
)

plt.axhline(
    0.90,
    linestyle="--",
    label="90% target"
)

plt.ylabel("Failure Coverage")
plt.title("Conformal Failure Coverage (LAC)")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "failure_coverage_lac.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. CONFORMAL AVERAGE SET SIZE
# ============================================================

plt.figure(figsize=(9, 5))

for method in ["LAC", "APS"]:

    subset = conformal[
        conformal["Method"] == method
    ]

    plt.plot(
        subset["Experiment"],
        subset["Average Set Size"],
        marker="o",
        label=method
    )

plt.ylabel("Average Prediction-Set Size")
plt.title("Conformal Prediction-Set Size")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "conformal_set_size.png",
    dpi=300
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("RESULT FIGURES CREATED")
print("=" * 60)

print(f"Folder: {FIGURES}")

print()

for file in sorted(FIGURES.glob("*.png")):
    print(file.name)