from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

WINDOW_DIR = (
    ROOT
    / "data"
    / "processed"
    / "standard"
    / "windows"
)

RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 4096
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

print("=" * 60)
print("STANDARD CNN TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):

    def __init__(self, x_path, y_path):
        self.X = np.load(
            x_path,
            mmap_mode="r"
        )

        self.y = np.load(
            y_path,
            mmap_mode="r"
        )

        print(
            f"Loaded X: {self.X.shape}"
        )

        print(
            f"Loaded y: {self.y.shape}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        # Copy the sample so PyTorch receives
        # a writable contiguous array.
        x = np.array(
            self.X[index],
            dtype=np.float32,
            copy=True
        )

        y = np.float32(
            self.y[index]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y)
        )


# ============================================================
# CNN
# ============================================================

class CNN1D(nn.Module):

    def __init__(self, n_features=16):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                n_features,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32, 1),
        )

    def forward(self, x):

        # Input:
        # (batch, 30, 16)

        # Conv1d expects:
        # (batch, features, sequence_length)

        x = x.transpose(1, 2)

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            probabilities.append(
                probs.cpu().numpy()
            )

            labels.append(
                y_batch.numpy()
            )

    probabilities = np.concatenate(
        probabilities
    )

    labels = np.concatenate(
        labels
    )

    roc = roc_auc_score(
        labels,
        probabilities
    )

    pr = average_precision_score(
        labels,
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probabilities": probabilities,
        "labels": labels,
    }


# ============================================================
# Load datasets
# ============================================================

train_dataset = WindowDataset(
    WINDOW_DIR / "train_X.npy",
    WINDOW_DIR / "train_y.npy"
)

cal_dataset = WindowDataset(
    WINDOW_DIR / "calibration_X.npy",
    WINDOW_DIR / "calibration_y.npy"
)

test_dataset = WindowDataset(
    WINDOW_DIR / "test_X.npy",
    WINDOW_DIR / "test_y.npy"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

cal_loader = DataLoader(
    cal_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Class imbalance
# ============================================================

positive_count = int(
    train_dataset.y.sum()
)

negative_count = int(
    len(train_dataset.y) - positive_count
)

pos_weight_value = (
    negative_count / positive_count
)

print()
print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(
    f"Positive windows: {positive_count:,}"
)

print(
    f"Negative windows: {negative_count:,}"
)

print(
    f"Positive rate: "
    f"{positive_count / len(train_dataset):.4%}"
)

print(
    f"Positive class weight: "
    f"{pos_weight_value:.2f}"
)


# ============================================================
# Model
# ============================================================

model = CNN1D(
    n_features=16
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=DEVICE
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_pr_auc = -1.0

best_model_path = (
    RESULT_DIR
    / "standard_cnn_best.pt"
)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    samples_seen = 0

    print()
    print("=" * 60)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 60)

    for batch_idx, (X_batch, y_batch) in enumerate(
        train_loader,
        start=1
    ):

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        batch_size = len(y_batch)

        running_loss += (
            loss.item() * batch_size
        )

        samples_seen += batch_size

        if batch_idx % 100 == 0:

            print(
                f"Batch {batch_idx:,} | "
                f"Samples {samples_seen:,} | "
                f"Loss "
                f"{running_loss / samples_seen:.5f}"
            )

    train_loss = (
        running_loss
        / samples_seen
    )

    # --------------------------------------------------------
    # Calibration evaluation
    # --------------------------------------------------------

    metrics = evaluate(
        model,
        cal_loader
    )

    print()
    print(
        f"Train loss: {train_loss:.6f}"
    )

    print(
        f"Calibration ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Calibration PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Calibration precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Calibration recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Calibration F1: "
        f"{metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Save best model according to PR-AUC
    # --------------------------------------------------------

    if metrics["pr_auc"] > best_pr_auc:

        best_pr_auc = metrics["pr_auc"]

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Saved best model "
            f"(PR-AUC={best_pr_auc:.4f})"
        )


# ============================================================
# Load best model
# ============================================================

print()
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)


# ============================================================
# Calibration predictions
# ============================================================

cal_metrics = evaluate(
    model,
    cal_loader
)

np.save(
    RESULT_DIR
    / "standard_cnn_calibration_probs.npy",
    cal_metrics["probabilities"]
)

np.save(
    RESULT_DIR
    / "standard_cnn_calibration_labels.npy",
    cal_metrics["labels"]
)


# ============================================================
# Test predictions
# ============================================================

test_metrics = evaluate(
    model,
    test_loader
)

np.save(
    RESULT_DIR
    / "standard_cnn_test_probs.npy",
    test_metrics["probabilities"]
)

np.save(
    RESULT_DIR
    / "standard_cnn_test_labels.npy",
    test_metrics["labels"]
)


# ============================================================
# Final results
# ============================================================

print()
print("=" * 60)
print("STANDARD CNN TEST RESULTS")
print("=" * 60)

print(
    f"ROC-AUC  : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC   : "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1       : "
    f"{test_metrics['f1']:.4f}"
)

print()
print(
    f"Best model: {best_model_path}"
)

print()
print("=" * 60)
print("STANDARD CNN TRAINING COMPLETE")
print("=" * 60)