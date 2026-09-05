from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

WINDOW_DIR = ROOT / "data" / "processed" / "dev" / "windows"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-3

torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("=" * 60)
print("SSD FAILURE PREDICTION — 1D CNN")
print("=" * 60)
print(f"Device: {DEVICE}")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_split(name):
    X = np.load(
        WINDOW_DIR / f"{name}_X.npy",
        mmap_mode="r"
    )
    y = np.load(
        WINDOW_DIR / f"{name}_y.npy"
    )

    print(
        f"{name:12}: "
        f"X={X.shape}, "
        f"positive={(y == 1).sum():,}, "
        f"negative={(y == 0).sum():,}"
    )

    return X, y


X_train, y_train = load_split("train")
X_cal, y_cal = load_split("calibration")
X_test, y_test = load_split("test")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class WindowDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = np.asarray(self.X[index], dtype=np.float32)
        y = np.float32(self.y[index])

        # CNN expects:
        # batch × channels × time
        x = torch.from_numpy(x).transpose(0, 1)

        return x, torch.tensor(y)


train_dataset = WindowDataset(X_train, y_train)
cal_dataset = WindowDataset(X_cal, y_cal)
test_dataset = WindowDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

cal_loader = DataLoader(
    cal_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

class SSDCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


model = SSDCNN().to(DEVICE)

# Handle class imbalance.
positive_count = (y_train == 1).sum()
negative_count = (y_train == 0).sum()

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


print()
print(f"Positive class weight: {pos_weight.item():.2f}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def predict(loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)

            logits = model(X)

            probs = torch.sigmoid(logits)

            probabilities.append(
                probs.cpu().numpy()
            )

            labels.append(
                y.numpy()
            )

    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
    )


def evaluate(name, loader):

    probabilities, labels = predict(loader)

    roc = roc_auc_score(
        labels,
        probabilities,
    )

    pr = average_precision_score(
        labels,
        probabilities,
    )

    print()
    print(f"{name}")
    print("-" * 40)
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr:.4f}")

    return probabilities, labels


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    batches = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        batches += 1

    avg_loss = running_loss / batches

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"| loss={avg_loss:.4f}"
    )

    # Evaluate on calibration set after every epoch.
    evaluate(
        "Calibration",
        cal_loader,
    )


# ------------------------------------------------------------
# Final evaluation
# ------------------------------------------------------------

print()
print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

cal_probs, cal_labels = evaluate(
    "CALIBRATION",
    cal_loader,
)

test_probs, test_labels = evaluate(
    "TEST",
    test_loader,
)


# ------------------------------------------------------------
# Save model and predictions
# ------------------------------------------------------------

model_path = RESULT_DIR / "cnn_baseline.pt"

torch.save(
    model.state_dict(),
    model_path,
)

np.save(
    RESULT_DIR / "cnn_calibration_probs.npy",
    cal_probs,
)

np.save(
    RESULT_DIR / "cnn_calibration_labels.npy",
    cal_labels,
)

np.save(
    RESULT_DIR / "cnn_test_probs.npy",
    test_probs,
)

np.save(
    RESULT_DIR / "cnn_test_labels.npy",
    test_labels,
)

print()
print("=" * 60)
print("CNN COMPLETE")
print("=" * 60)
print(f"Model saved: {model_path}")