from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

WINDOW_ROOT = Path(
    "data/processed/lomo/windows"
)

RESULT_ROOT = Path(
    "results/lomo"
)

LOMO_FOLDS = ["A", "B", "C"]

BATCH_SIZE = 4096
EPOCHS = 5
LEARNING_RATE = 1e-3

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# MODEL
# ============================================================

class CNN1D(nn.Module):

    def __init__(self, n_features):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=n_features,
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
        # (batch, 30, features)

        # Conv1d expects:
        # (batch, features, 30)

        x = x.transpose(1, 2)

        x = self.features(x)

        return self.classifier(x).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(lomo, split):

    split_dir = (
        WINDOW_ROOT
        / f"lomo_{lomo}"
        / split
    )

    X_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"

    if not X_path.exists():
        raise FileNotFoundError(
            f"Missing X file:\n{X_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Missing y file:\n{y_path}"
        )

    # mmap keeps the huge arrays from being unnecessarily
    # copied into memory.
    X = np.load(
        X_path,
        mmap_mode="r",
    )

    y = np.load(y_path)

    if X.ndim != 3:
        raise ValueError(
            f"{X_path} has shape {X.shape}; "
            f"expected 3 dimensions."
        )

    if X.shape[0] != len(y):
        raise ValueError(
            f"X/y mismatch for {lomo}/{split}: "
            f"{X.shape[0]} vs {len(y)}"
        )

    if X.shape[1] != 30:
        raise ValueError(
            f"Expected window length 30, "
            f"got {X.shape[1]}"
        )

    return X, y


# ============================================================
# CREATE DATA LOADER
# ============================================================

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        # .copy() avoids the non-writable NumPy warning.
        x = np.array(
            self.X[idx],
            dtype=np.float32,
            copy=True,
        )

        y = np.float32(
            self.y[idx]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y),
        )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def predict(model, X, batch_size):

    model.eval()

    probabilities = []

    n = len(X)

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        batch = np.array(
            X[start:end],
            dtype=np.float32,
            copy=True,
        )

        batch = torch.from_numpy(
            batch
        ).to(DEVICE)

        logits = model(batch)

        probs = torch.sigmoid(
            logits
        )

        probabilities.append(
            probs.cpu().numpy()
        )

    return np.concatenate(
        probabilities
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TRAIN ONE LOMO FOLD
# ============================================================

def train_lomo(lomo):

    print()
    print("=" * 70)
    print(f"LOMO-{lomo}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, y_train = load_split(
        lomo,
        "train",
    )

    X_cal, y_cal = load_split(
        lomo,
        "calibration",
    )

    X_test, y_test = load_split(
        lomo,
        "test",
    )

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Calibration: {X_cal.shape}"
    )

    print(
        f"Test: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Determine number of features directly from data.
    #
    # This is IMPORTANT.
    #
    # We do NOT hard-code 15 or 16 here.
    # --------------------------------------------------------

    n_features = X_train.shape[2]

    if X_cal.shape[2] != n_features:
        raise ValueError(
            "Train/calibration feature mismatch: "
            f"{n_features} vs {X_cal.shape[2]}"
        )

    if X_test.shape[2] != n_features:
        raise ValueError(
            "Train/test feature mismatch: "
            f"{n_features} vs {X_test.shape[2]}"
        )

    print(
        f"Features: {n_features}"
    )

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    positives = int(
        (y_train == 1).sum()
    )

    negatives = int(
        (y_train == 0).sum()
    )

    if positives == 0:
        raise ValueError(
            f"LOMO-{lomo} training set "
            "contains no positive samples."
        )

    pos_weight = (
        negatives / positives
    )

    print(
        f"Training positives: "
        f"{positives:,}"
    )

    print(
        f"Training negatives: "
        f"{negatives:,}"
    )

    print(
        f"Positive weight: "
        f"{pos_weight:.2f}"
    )

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = NumpyDataset(
        X_train,
        y_train,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNN1D(
        n_features=n_features
    ).to(DEVICE)

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

    # --------------------------------------------------------
    # Best model selected by calibration PR-AUC.
    # --------------------------------------------------------

    best_pr_auc = -np.inf

    best_state = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        total_samples = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                y_batch.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        average_loss = (
            total_loss
            / total_samples
        )

        # ----------------------------------------------------
        # Calibration evaluation
        # ----------------------------------------------------

        cal_probs = predict(
            model,
            X_cal,
            BATCH_SIZE,
        )

        cal_metrics = calculate_metrics(
            y_cal,
            cal_probs,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss {average_loss:.6f} | "
            f"cal ROC-AUC "
            f"{cal_metrics['roc_auc']:.4f} | "
            f"cal PR-AUC "
            f"{cal_metrics['pr_auc']:.4f} | "
            f"precision "
            f"{cal_metrics['precision']:.4f} | "
            f"recall "
            f"{cal_metrics['recall']:.4f} | "
            f"F1 "
            f"{cal_metrics['f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model.
        # ----------------------------------------------------

        if (
            cal_metrics["pr_auc"]
            > best_pr_auc
        ):

            best_pr_auc = (
                cal_metrics["pr_auc"]
            )

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            print(
                f"  -> New best "
                f"calibration PR-AUC: "
                f"{best_pr_auc:.4f}"
            )

    # --------------------------------------------------------
    # Restore best model.
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No best model was saved."
        )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Final calibration probabilities
    # --------------------------------------------------------

    cal_probs = predict(
        model,
        X_cal,
        BATCH_SIZE,
    )

    cal_metrics = calculate_metrics(
        y_cal,
        cal_probs,
    )

    # --------------------------------------------------------
    # Final test probabilities
    # --------------------------------------------------------

    test_probs = predict(
        model,
        X_test,
        BATCH_SIZE,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probs,
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        f"LOMO-{lomo} FINAL "
        "CALIBRATION"
    )

    print(
        f"ROC-AUC   : "
        f"{cal_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{cal_metrics['pr_auc']:.4f}"
    )

    print(
        f"Precision : "
        f"{cal_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{cal_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{cal_metrics['f1']:.4f}"
    )

    print()
    print(
        f"LOMO-{lomo} FINAL TEST"
    )

    print(
        f"ROC-AUC   : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{test_metrics['f1']:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        RESULT_ROOT
        / f"lomo_{lomo}_cnn_best.pt"
    )

    cal_probs_path = (
        RESULT_ROOT
        / f"lomo_{lomo}_calibration_probs.npy"
    )

    cal_labels_path = (
        RESULT_ROOT
        / f"lomo_{lomo}_calibration_labels.npy"
    )

    test_probs_path = (
        RESULT_ROOT
        / f"lomo_{lomo}_test_probs.npy"
    )

    test_labels_path = (
        RESULT_ROOT
        / f"lomo_{lomo}_test_labels.npy"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    np.save(
        cal_probs_path,
        cal_probs,
    )

    np.save(
        cal_labels_path,
        y_cal,
    )

    np.save(
        test_probs_path,
        test_probs,
    )

    np.save(
        test_labels_path,
        y_test,
    )

    print()
    print("Saved:")

    print(
        f"  {model_path}"
    )

    print(
        f"  {cal_probs_path}"
    )

    print(
        f"  {cal_labels_path}"
    )

    print(
        f"  {test_probs_path}"
    )

    print(
        f"  {test_labels_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for lomo in LOMO_FOLDS:

        train_lomo(lomo)

    print()
    print("=" * 70)
    print("ALL LOMO CNN EXPERIMENTS COMPLETE")
    print("=" * 70)