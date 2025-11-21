"""
Dataset tests
"""

import numpy as np
import pytest

import src.dataset_asl_mnist as dsm
from src.dataset_asl_mnist import normalize_landmarks, ASLMNISTDataset


# -----------------------------
# normalize_landmarks tests
# -----------------------------
def test_normalize_landmarks_translates_and_scales():
    """
    Wrist should be moved to the origin and the farthest point
    should have distance 1 after normalization.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    pts[0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pts[1] = np.array([3.0, 2.0, 3.0], dtype=np.float32)

    norm = normalize_landmarks(pts)
    norm_pts = norm.reshape(21, 3)

    assert np.allclose(norm_pts[0], np.zeros(3, dtype=np.float32))

    dists = np.linalg.norm(norm_pts, axis=1)
    assert np.isclose(dists.max(), 1.0, atol=1e-6)


# -----------------------------
# load_asl_mnist_with_retries tests
# -----------------------------
def test_load_asl_mnist_with_retries_handles_429(monkeypatch):
    """
    If load_dataset raises a 429 error first, the function should retry
    and eventually return the dataset.
    """
    calls = {"n": 0}

    def fake_load_dataset(name, split):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("HTTP 429 Too Many Requests")
        return ["dummy-dataset"]

    monkeypatch.setattr(dsm, "load_dataset", fake_load_dataset)

    monkeypatch.setattr(dsm.time, "sleep", lambda s: None)

    result = dsm.load_asl_mnist_with_retries(
        split="train",
        max_retries=3,
        base_delay=1,
    )

    assert result == ["dummy-dataset"]
    assert calls["n"] == 2


def test_load_asl_mnist_with_retries_raises_on_other_errors(monkeypatch):
    """
    Non-429 errors should be raised immediately (no retry loop).
    """

    def fake_load_dataset(name, split):
        raise Exception("Some other error")

    monkeypatch.setattr(dsm, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(dsm.time, "sleep", lambda s: None)

    with pytest.raises(Exception) as excinfo:
        dsm.load_asl_mnist_with_retries(split="train", max_retries=3, base_delay=1)

    assert "Some other error" in str(excinfo.value)


# -----------------------------
# ASLMNISTDataset tests
# -----------------------------


class DummyHFDS:
    """Minimal fake HF dataset with image + label fields."""

    def __init__(self):
        self._samples = [
            {"image": np.zeros((28, 28, 3), dtype=np.uint8), "label": 10},
            {"image": np.ones((28, 28, 3), dtype=np.uint8), "label": 11},
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def test_aslmnistdataset_getitem_with_landmarks(monkeypatch):
    """
    When _extract_landmarks returns valid (21,3) pts, __getitem__
    should return a (63,) normalized vector and mapped label.
    """
    monkeypatch.setattr(
        dsm, "load_asl_mnist_with_retries", lambda split="train": DummyHFDS()
    )

    def fake_load_label_maps(label_map_path=None):
        index_to_letter = {0: "A", 1: "B"}
        letter_to_index = {"A": 0, "B": 1}
        raw_label_to_index = {10: 0, 11: 1}
        return index_to_letter, letter_to_index, raw_label_to_index

    monkeypatch.setattr(dsm, "load_label_maps", fake_load_label_maps)

    def fake_extract_landmarks(self, pil_img):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[1] = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        return pts

    monkeypatch.setattr(
        ASLMNISTDataset, "_extract_landmarks", fake_extract_landmarks, raising=False
    )

    ds = ASLMNISTDataset(split="train")

    assert len(ds) == 2

    X, label = ds[0]
    assert isinstance(X, np.ndarray)
    assert X.shape == (63,)
    assert label in (0, 1)

    pts = X.reshape(21, 3)
    dists = np.linalg.norm(pts, axis=1)
    assert np.isclose(dists.max(), 1.0, atol=1e-6)


def test_aslmnistdataset_getitem_with_no_landmarks(monkeypatch):
    """
    If _extract_landmarks returns None, __getitem__ should return
    a zero vector of shape (63,).
    """
    monkeypatch.setattr(
        dsm, "load_asl_mnist_with_retries", lambda split="train": DummyHFDS()
    )

    def fake_load_label_maps(label_map_path=None):
        index_to_letter = {0: "A", 1: "B"}
        letter_to_index = {"A": 0, "B": 1}
        raw_label_to_index = {10: 0, 11: 1}
        return index_to_letter, letter_to_index, raw_label_to_index

    monkeypatch.setattr(dsm, "load_label_maps", fake_load_label_maps)

    monkeypatch.setattr(
        ASLMNISTDataset,
        "_extract_landmarks",
        lambda self, pil_img: None,
        raising=False,
    )

    ds = ASLMNISTDataset(split="train")

    X, label = ds[0]
    assert X.shape == (63,)
    assert np.allclose(X, np.zeros((63,), dtype=np.float32))
    assert label in (0, 1)
