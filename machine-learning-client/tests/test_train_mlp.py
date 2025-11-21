from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import src.train_mlp as tm


def test_load_dataset_returns_tensor_dataset(tmp_path, monkeypatch):
    fake_npz = tmp_path / "webcam_landmarks.npz"
    X = np.random.randn(10, 63).astype(np.float32)
    y = np.random.randint(0, 5, size=(10,))
    np.savez(fake_npz, X=X, y=y)

    monkeypatch.setattr(tm, "DATA_PATH", fake_npz)

    ds = tm.load_dataset()

    assert isinstance(ds, TensorDataset)
    assert len(ds) == 10
    X_t, y_t = ds.tensors
    assert X_t.shape == (10, 63)
    assert y_t.shape == (10,)
    assert X_t.dtype == torch.float32
    assert y_t.dtype == torch.int64


def test_train_runs_and_saves_best_model(tmp_path, monkeypatch):
    # Make an "easy" dataset: all labels are 0
    X = torch.randn(40, 63)
    y = torch.zeros(40, dtype=torch.long)
    fake_dataset = TensorDataset(X, y)

    monkeypatch.setattr(tm, "load_dataset", lambda: fake_dataset)

    out_file = tmp_path / "mlp_webcam.pt"
    monkeypatch.setattr(tm, "OUT_PATH", out_file)

    mock_save = MagicMock()
    monkeypatch.setattr(tm.torch, "save", mock_save)

    tm.train(num_epochs=1, batch_size=8, val_split=0.25)

    assert mock_save.call_count >= 1


def test_train_detects_correct_num_classes(monkeypatch, tmp_path):
    # labels {0,1,2}
    X = torch.randn(30, 63)
    y = torch.tensor([0, 1, 2] * 10, dtype=torch.long)
    fake_dataset = TensorDataset(X, y)
    monkeypatch.setattr(tm, "load_dataset", lambda: fake_dataset)

    out_file = tmp_path / "mlp_webcam.pt"
    monkeypatch.setattr(tm, "OUT_PATH", out_file)

    captured = {"num_classes": None}

    class CapturingModel(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            captured["num_classes"] = num_classes
            self.linear = torch.nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.linear(x)

    monkeypatch.setattr(tm, "LandmarkMLP", CapturingModel)
    monkeypatch.setattr(tm.torch, "save", lambda *args, **kwargs: None)

    tm.train(num_epochs=0, batch_size=8, val_split=0.2)

    assert captured["num_classes"] == 3
