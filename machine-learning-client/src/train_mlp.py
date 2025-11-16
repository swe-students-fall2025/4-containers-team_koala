from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from models.model_MLP import LandmarkMLP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "webcam_landmarks.npz"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = MODELS_DIR / "mlp_webcam.pt"


def load_dataset():
    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]

    print(f"Loaded landmarks: X shape={X.shape}, y shape={y.shape}")

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    return TensorDataset(X_t, y_t)


def train(
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 30,
    val_split: float = 0.2,
):

    dataset = load_dataset()
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Infer num_classes from labels
    all_labels = torch.unique(dataset.tensors[1])
    num_classes = int(all_labels.max().item()) + 1
    print(f"Detected {num_classes} classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkMLP(input_dim=63, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_ds)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  ▶ New best model saved to {OUT_PATH} (val acc: {best_val_acc:.4f})")

    print(f"Training done. Best val acc: {best_val_acc:.4f}")
    print(f"Final model weights at: {OUT_PATH}")


if __name__ == "__main__":
    train()
