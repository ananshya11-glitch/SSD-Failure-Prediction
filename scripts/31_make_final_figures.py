from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

INPUT = REPORTS / "master_results.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(f"Missing: {INPUT}")

df = pd.read_csv(INPUT)

required = [
    "setting",
    "roc_auc",
    "pr_auc",
    "drive_coverage",
    "drive_failure_coverage",
    "drive_avg_set_size",
    "method",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# FIGURE 1 — CNN ROC-AUC
# ============================================================

cnn = (
    df[
        ["setting", "roc_auc"]
    ]
    .drop_duplicates("setting")
)

plt.figure(figsize=(8, 5))

plt.bar(
    cnn["setting"],
    cnn["roc_auc"],
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

plt.ylabel("ROC-AUC")
plt.xlabel("Evaluation Setting")
plt.title("CNN ROC-AUC Across Evaluation Settings")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    FIGURES / "figure1_cnn_roc_auc.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 2 — CNN PR-AUC
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    cnn["setting"],
    df[
        ["setting", "pr_auc"]
    ]
    .drop_duplicates("setting")["pr_auc"],
)

plt.ylabel("PR-AUC")
plt.xlabel("Evaluation Setting")
plt.title("CNN PR-AUC Across Evaluation Settings")

plt.tight_layout()

plt.savefig(
    FIGURES / "figure2_cnn_pr_auc.png",
    dpi=300,
)

plt.close()


# ============================================================
# LOMO CONFORMAL RESULTS
# ============================================================

lomo = df[
    df["setting"].isin(
        ["LOMO-A", "LOMO-B", "LOMO-C"]
    )
].copy()


# ============================================================
# FIGURE 3 — OVERALL COVERAGE
# ============================================================

pivot = lomo.pivot(
    index="setting",
    columns="method",
    values="drive_coverage",
)

plt.figure(figsize=(8, 5))

pivot.plot(
    kind="bar",
    ax=plt.gca(),
)

plt.axhline(
    0.90,
    linestyle="--",
    linewidth=1,
    label="Target coverage = 90%",
)

plt.ylabel("Coverage")
plt.xlabel("LOMO Setting")
plt.title("Drive-Level Conformal Coverage Under Vendor Shift")

plt.ylim(0, 1.05)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "figure3_conformal_coverage.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 4 — FAILURE COVERAGE
# ============================================================

pivot = lomo.pivot(
    index="setting",
    columns="method",
    values="drive_failure_coverage",
)

plt.figure(figsize=(8, 5))

pivot.plot(
    kind="bar",
    ax=plt.gca(),
)

plt.axhline(
    0.90,
    linestyle="--",
    linewidth=1,
    label="Target = 90%",
)

plt.ylabel("Failure Coverage")
plt.xlabel("LOMO Setting")
plt.title("Conformal Failure Coverage Under Vendor Shift")

plt.ylim(0, 1.05)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "figure4_failure_coverage.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 5 — AVERAGE PREDICTION-SET SIZE
# ============================================================

pivot = lomo.pivot(
    index="setting",
    columns="method",
    values="drive_avg_set_size",
)

plt.figure(figsize=(8, 5))

pivot.plot(
    kind="bar",
    ax=plt.gca(),
)

plt.ylabel("Average Prediction-Set Size")
plt.xlabel("LOMO Setting")
plt.title("Conformal Prediction-Set Size Under Vendor Shift")

plt.ylim(0, 2.1)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "figure5_prediction_set_size.png",
    dpi=300,
)

plt.close()


# ============================================================
# SAVE A CLEAN PAPER TABLE
# ============================================================

paper_table = df[
    [
        "setting",
        "method",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "drive_coverage",
        "drive_failure_coverage",
        "drive_avg_set_size",
    ]
].copy()

paper_table.to_csv(
    FIGURES / "paper_results_table.csv",
    index=False,
)


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("FINAL FIGURES CREATED")
print("=" * 70)

print()

for path in sorted(FIGURES.iterdir()):
    print(path.name)

print()
print(f"Saved to: {FIGURES}")