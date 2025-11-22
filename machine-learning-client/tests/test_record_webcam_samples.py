# tests/test_record_webcam_samples.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import src.record_webcam_samples as rws


# ---------------------------------------------------------
# Test: load_letter_to_index
# ---------------------------------------------------------
def test_load_letter_to_index(tmp_path, monkeypatch):
    fake_map = {"letter_to_index": {"A": 0, "B": 1}}
    fake_file = tmp_path / "label_map.json"
    fake_file.write_text(json_dump := '{"letter_to_index": {"A": 0, "B": 1}}')

    monkeypatch.setattr(rws, "LABEL_MAP_PATH", fake_file)

    mapping = rws.load_letter_to_index()
    assert mapping == {"A": 0, "B": 1}


# ---------------------------------------------------------
# Test: append_to_npz (creating new file)
# ---------------------------------------------------------
def test_append_to_npz_creates_file(tmp_path):
    out = tmp_path / "dataset.npz"

    X_new = np.random.randn(5, 63)
    y_new = np.array([1, 1, 1, 1, 1])

    rws.append_to_npz(out, X_new, y_new)

    data = np.load(out)
    assert data["X"].shape == (5, 63)
    assert data["y"].shape == (5,)
    np.testing.assert_allclose(data["X"], X_new)
    np.testing.assert_allclose(data["y"], y_new)


# ---------------------------------------------------------
# Test: append_to_npz (appending to existing file)
# ---------------------------------------------------------
def test_append_to_npz_appends(tmp_path):
    out = tmp_path / "dataset.npz"

    X_old = np.random.randn(3, 63)
    y_old = np.array([0, 0, 0])
    np.savez(out, X=X_old, y=y_old)

    X_new = np.random.randn(2, 63)
    y_new = np.array([1, 1])

    rws.append_to_npz(out, X_new, y_new)

    data = np.load(out)
    X_combined = np.concatenate([X_old, X_new])
    y_combined = np.concatenate([y_old, y_new])

    np.testing.assert_allclose(data["X"], X_combined)
    np.testing.assert_allclose(data["y"], y_combined)


# ---------------------------------------------------------
# Test: main() → ensures recording loop saves NPZ correctly
# ---------------------------------------------------------
def test_main_saves_samples(tmp_path, monkeypatch):
    fake_map = tmp_path / "label_map.json"
    fake_map.write_text('{"letter_to_index": {"A": 0}}')
    monkeypatch.setattr(rws, "LABEL_MAP_PATH", fake_map)

    out_npz = tmp_path / "webcam_landmarks.npz"
    monkeypatch.setattr(rws, "OUT_PATH", out_npz)

    mock_cap = MagicMock()
    mock_cap.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None),
    ]
    monkeypatch.setattr(rws.cv2, "VideoCapture", lambda _: mock_cap)

    monkeypatch.setattr(rws.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(rws.cv2, "destroyAllWindows", lambda: None)

    monkeypatch.setattr(rws.cv2, "waitKey", lambda _: ord("q"))

    monkeypatch.setattr(
        rws, "normalize_landmarks", lambda pts: np.ones(63, dtype=np.float32)
    )

    rws.main()

    assert out_npz.exists()
    data = np.load(out_npz)
    assert data["X"].shape[1] == 63
    assert data["y"].shape == (data["X"].shape[0],)
