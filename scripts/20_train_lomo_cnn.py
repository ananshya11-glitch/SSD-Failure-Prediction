from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# CONFIG
# ============================================================

WINDOW_ROOT = Path("data/processed/lomo/windows")
RESULTS = Path("results/lomo")

BATCH_SIZE = 4096
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

LOMO_CONFIG = {
    "A": ("train", "calibration", "test"),
    "B": ("train", "calibration", "test"),
    "C": ("train", "calibration", "test"),
}


# ============================================================
# CNN
# ============================================================

class CNN1D(nn.Module):

    def __init__(self, n_features=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

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
        # x = batch × time × features
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(lomo, split):

    path = WINDOW_ROOT / f"lomo_{lomo}" / split

    X = np.load(path / "X.npy", mmap_mode="r")
    y = np.load(path / "y.npy")

    return X, y


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X, y):

    model.eval()

    probabilities = []

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(X)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.float().to(DEVICE)

            logits = model(batch)

            probs = torch.sigmoid(logits)

            probabilities.append(
                probs.cpu().numpy()
            )

    probabilities = np.concatenate(probabilities)

    roc = roc_auc_score(y, probabilities)
    pr = average_precision_score(y, probabilities)

    predictions = (probabilities >= 0.5).astype(np.int8)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    return probabilities, roc, pr, precision, recall, f1


# ============================================================
# TRAIN ONE LOMO MODEL
# ============================================================

def train_lomo(lomo):

    print("\n" + "=" * 70)
    print(f"LOMO-{lomo}")
    print("=" * 70)

    X_train, y_train = load_split(lomo, "train")
    X_cal, y_cal = load_split(lomo, "calibration")
    X_test, y_test = load_split(lomo, "test")

    print(f"Train: {X_train.shape}")
    print(f"Calibration: {X_cal.shape}")
    print(f"Test: {X_test.shape}")

    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())

    pos_weight = negatives / positives

    print(f"Training positives: {positives:,}")
    print(f"Training negatives: {negatives:,}")
    print(f"Positive weight: {pos_weight:.2f}")

    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_train)),
        torch.from_numpy(y_train.astype(np.float32)),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = CNN1D().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_pr = -1

    RESULTS.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS / f"lomo_{lomo}_best.pt"

    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        batches = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.float().to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches

        (
            cal_probs,
            cal_roc,
            cal_pr,
            cal_precision,
            cal_recall,
            cal_f1,
        ) = evaluate(
            model,
            X_cal,
            y_cal
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Loss: {avg_loss:.4f}"
        )

        print(
            f"Calibration ROC-AUC: {cal_roc:.4f}"
        )

        print(
            f"Calibration PR-AUC: {cal_pr:.4f}"
        )

        print(
            f"Precision: {cal_precision:.4f} | "
            f"Recall: {cal_recall:.4f} | "
            f"F1: {cal_f1:.4f}"
        )

        # Save best checkpoint based on calibration PR-AUC
        if cal_pr > best_pr:

            best_pr = cal_pr

            torch.save(
                model.state_dict(),
                model_path
            )

            np.save(
                RESULTS /
                f"lomo_{lomo}_calibration_probs.npy",
                cal_probs
            )

            np.save(
                RESULTS /
                f"lomo_{lomo}_calibration_labels.npy",
                y_cal
            )

            print("Saved BEST model.")

    # ========================================================
    # TEST USING BEST MODEL
    # ========================================================

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    (
        test_probs,
        test_roc,
        test_pr,
        test_precision,
        test_recall,
        test_f1,
    ) = evaluate(
        model,
        X_test,
        y_test
    )

    np.save(
        RESULTS /
        f"lomo_{lomo}_test_probs.npy",
        test_probs
    )

    np.save(
        RESULTS /
        f"lomo_{lomo}_test_labels.npy",
        y_test
    )

    print("\n" + "-" * 50)
    print(f"FINAL LOMO-{lomo} TEST RESULTS")
    print("-" * 50)

    print(f"ROC-AUC : {test_roc:.4f}")
    print(f"PR-AUC  : {test_pr:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1       : {test_f1:.4f}")

    print(
        f"\nSaved model: {model_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    for lomo in ["A", "B", "C"]:

        train_lomo(lomo)

    print("\n" + "=" * 70)
    print("ALL LOMO CNN EXPERIMENTS COMPLETE")
    print("=" * 70)