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
OUT = ROOT / "reports"

OUT.mkdir(exist_ok=True)


# ============================================================
# CNN RESULTS
# ============================================================

experiments = []


def add_experiment(name, probs_path, labels_path):

    probs = np.load(probs_path)
    labels = np.load(labels_path)

    predictions = (
        probs >= 0.5
    ).astype(np.int8)

    experiments.append({
        "Experiment": name,
        "ROC-AUC": roc_auc_score(labels, probs),
        "PR-AUC": average_precision_score(labels, probs),
        "Precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "Test Windows": len(labels),
        "Positive Windows": int(labels.sum()),
        "Positive Rate": labels.mean(),
    })


# STANDARD

add_experiment(
    "STANDARD",
    RESULTS / "standard_cnn_test_probs.npy",
    RESULTS / "standard_cnn_test_labels.npy",
)


# LOMO

for lomo in ["A", "B", "C"]:

    add_experiment(
        f"LOMO-{lomo}",
        RESULTS / "lomo" / f"lomo_{lomo}_test_probs.npy",
        RESULTS / "lomo" / f"lomo_{lomo}_test_labels.npy",
    )


# ============================================================
# CNN TABLE
# ============================================================

df = pd.DataFrame(experiments)

print()
print("=" * 100)
print("CNN TEST RESULTS")
print("=" * 100)

print(
    df.to_string(
        index=False,
        formatters={
            "ROC-AUC": "{:.4f}".format,
            "PR-AUC": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "Positive Rate": "{:.6%}".format,
        },
    )
)


# ============================================================
# CONFORMAL RESULTS
# ============================================================

# These values come directly from the completed conformal
# experiments.

conformal = pd.DataFrame([
    {
        "Experiment": "STANDARD",
        "Method": "LAC",
        "Coverage": 0.9030,
        "Healthy Coverage": 0.9029,
        "Failure Coverage": 0.9591,
        "Average Set Size": 1.6839,
    },
    {
        "Experiment": "STANDARD",
        "Method": "APS",
        "Coverage": 1.0000,
        "Healthy Coverage": 1.0000,
        "Failure Coverage": 1.0000,
        "Average Set Size": 2.0000,
    },

    {
        "Experiment": "LOMO-A",
        "Method": "LAC",
        "Coverage": 0.9957,
        "Healthy Coverage": 0.9963,
        "Failure Coverage": 0.0000,
        "Average Set Size": 1.0000,
    },
    {
        "Experiment": "LOMO-A",
        "Method": "APS",
        "Coverage": 0.9940,
        "Healthy Coverage": 0.9946,
        "Failure Coverage": 0.0000,
        "Average Set Size": 0.9946,
    },

    {
        "Experiment": "LOMO-B",
        "Method": "LAC",
        "Coverage": 0.9991,
        "Healthy Coverage": 1.0000,
        "Failure Coverage": 0.0000,
        "Average Set Size": 1.0000,
    },
    {
        "Experiment": "LOMO-B",
        "Method": "APS",
        "Coverage": 0.9950,
        "Healthy Coverage": 0.9959,
        "Failure Coverage": 0.0000,
        "Average Set Size": 0.9959,
    },

    {
        "Experiment": "LOMO-C",
        "Method": "LAC",
        "Coverage": 0.3942,
        "Healthy Coverage": 0.3927,
        "Failure Coverage": 0.9184,
        "Average Set Size": 1.3401,
    },
    {
        "Experiment": "LOMO-C",
        "Method": "APS",
        "Coverage": 1.0000,
        "Healthy Coverage": 1.0000,
        "Failure Coverage": 1.0000,
        "Average Set Size": 2.0000,
    },
])


# ============================================================
# SAVE
# ============================================================

cnn_path = OUT / "cnn_results.csv"
conformal_path = OUT / "conformal_results.csv"

df.to_csv(
    cnn_path,
    index=False,
)

conformal.to_csv(
    conformal_path,
    index=False,
)


# ============================================================
# PRINT CONFORMAL TABLE
# ============================================================

print()
print("=" * 100)
print("CONFORMAL RESULTS")
print("=" * 100)

print(
    conformal.to_string(
        index=False,
        formatters={
            "Coverage": "{:.4f}".format,
            "Healthy Coverage": "{:.4f}".format,
            "Failure Coverage": "{:.4f}".format,
            "Average Set Size": "{:.4f}".format,
        },
    )
)


print()
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(cnn_path)
print(conformal_path)