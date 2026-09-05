from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WINDOW_DIR = ROOT / "data" / "processed" / "dev" / "windows"


def load_split(name):
    X = np.load(WINDOW_DIR / f"{name}_X.npy", mmap_mode="r")
    y = np.load(WINDOW_DIR / f"{name}_y.npy")

    print(f"{name}: X={X.shape}, y={y.shape}")

    return X, y


def make_features(X):
    """
    Convert each (30, 16) window into statistical features.

    For every SMART feature:
      - last value
      - mean
      - standard deviation
      - minimum
      - maximum
      - linear trend
    """

    n_samples = X.shape[0]
    n_features = X.shape[2]

    result = np.empty(
        (n_samples, n_features * 6),
        dtype=np.float32,
    )

    # Time axis: 0 ... 29
    t = np.arange(X.shape[1], dtype=np.float32)
    t_mean = t.mean()

    denominator = np.sum((t - t_mean) ** 2)

    for j in range(n_features):

        values = X[:, :, j]

        last = values[:, -1]
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        minimum = values.min(axis=1)
        maximum = values.max(axis=1)

        # Linear trend / slope
        slope = np.sum(
            (values - mean[:, None]) * (t - t_mean),
            axis=1,
        ) / denominator

        start = j * 6

        result[:, start] = last
        result[:, start + 1] = mean
        result[:, start + 2] = std
        result[:, start + 3] = minimum
        result[:, start + 4] = maximum
        result[:, start + 5] = slope

    return result


print("=" * 60)
print("SSD FAILURE PREDICTION — BASELINE")
print("=" * 60)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X_train, y_train = load_split("train")
X_cal, y_cal = load_split("calibration")
X_test, y_test = load_split("test")

# ------------------------------------------------------------
# Convert windows into statistical features
# ------------------------------------------------------------

print("\nExtracting statistical features...")

X_train_feat = make_features(X_train)
X_cal_feat = make_features(X_cal)
X_test_feat = make_features(X_test)

print(f"Feature shape: {X_train_feat.shape}")

# ------------------------------------------------------------
# Normalize using TRAINING data only
# ------------------------------------------------------------

print("\nFitting scaler on training data...")

scaler = StandardScaler()

X_train_feat = scaler.fit_transform(X_train_feat)
X_cal_feat = scaler.transform(X_cal_feat)
X_test_feat = scaler.transform(X_test_feat)

# ------------------------------------------------------------
# Train logistic regression
# ------------------------------------------------------------

print("\nTraining logistic regression...")

model = LogisticRegression(
    max_iter=300,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)

model.fit(X_train_feat, y_train)

print("Training complete.")

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

def evaluate(name, X, y):

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    roc = roc_auc_score(y, probabilities)
    pr = average_precision_score(y, probabilities)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification report:")
    print(classification_report(y, predictions, digits=4))

    return probabilities


cal_probs = evaluate(
    "CALIBRATION",
    X_cal_feat,
    y_cal,
)

test_probs = evaluate(
    "TEST",
    X_test_feat,
    y_test,
)

print()
print("=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)