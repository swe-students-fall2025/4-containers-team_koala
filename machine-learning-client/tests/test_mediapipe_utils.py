# tests/test_mediapipe_utils.py

import numpy as np
import pytest

import src.mediapipe_utils as mpu
from src.mediapipe_utils import HandLandmarks, normalize_landmarks, MediaPipeHandDetector


def test_normalize_landmarks_basic():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pts[1] = np.array([3.0, 2.0, 3.0], dtype=np.float32)

    norm = normalize_landmarks(pts)
    norm_pts = norm.reshape(21, 3)

    assert norm.shape == (63,)
    assert np.allclose(norm_pts[0], np.zeros(3, dtype=np.float32))
    dists = np.linalg.norm(norm_pts, axis=1)
    assert np.isclose(dists.max(), 1.0, atol=1e-6)


def test_normalize_landmarks_all_same_point():
    pts = np.ones((21, 3), dtype=np.float32)
    norm = normalize_landmarks(pts)
    norm_pts = norm.reshape(21, 3)

    assert norm.shape == (63,)
    assert np.allclose(norm_pts, np.zeros_like(norm_pts))


def test_draw_hand_landmarks_on_frame(monkeypatch):
    h, w = 100, 200
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    pts = np.stack([np.linspace(0, 1, 21), np.linspace(0, 1, 21), np.zeros(21)], axis=1)
    hand = HandLandmarks(points=pts.astype(np.float32), handedness="Right")

    calls = {"line": 0, "circle": 0, "text": 0}

    def fake_line(img, p1, p2, color, thickness):
        calls["line"] += 1
        return img

    def fake_circle(img, center, radius, color, thickness):
        calls["circle"] += 1
        return img

    def fake_puttext(img, text, org, font, font_scale, color, thickness):
        calls["text"] += 1
        return img

    monkeypatch.setattr(mpu.cv2, "line", fake_line)
    monkeypatch.setattr(mpu.cv2, "circle", fake_circle)
    monkeypatch.setattr(mpu.cv2, "putText", fake_puttext)

    mpu.draw_hand_landmarks_on_frame(frame, hand)

    assert calls["circle"] == 21
    assert calls["line"] >= 1
    assert calls["text"] == 1


def test_mediapipe_hand_detector_process(monkeypatch):
    class FakeLm:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeHandLandmarks:
        def __init__(self):
            self.landmark = [FakeLm(0.1 * i, 0.01 * i, 0.0) for i in range(21)]

    class FakeClassification:
        def __init__(self, label):
            self.label = label

    class FakeHandedness:
        def __init__(self, label):
            self.classification = [FakeClassification(label)]

    class FakeResults:
        def __init__(self):
            self.multi_hand_landmarks = [FakeHandLandmarks()]
            self.multi_handedness = [FakeHandedness("Right")]

    class FakeHands:
        def __init__(self, static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence):
            self.called_with = {
                "static_image_mode": static_image_mode,
                "max_num_hands": max_num_hands,
                "min_detection_confidence": min_detection_confidence,
                "min_tracking_confidence": min_tracking_confidence,
            }

        def process(self, frame_rgb):
            return FakeResults()

        def close(self):
            pass

    monkeypatch.setattr(mpu.mp_hands, "Hands", FakeHands)

    detector = MediaPipeHandDetector(max_num_hands=1, detection_confidence=0.5, tracking_confidence=0.5)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hands = detector.process(frame)

    assert len(hands) == 1
    hand = hands[0]
    assert isinstance(hand, HandLandmarks)
    assert hand.points.shape == (21, 3)
    assert hand.points.dtype == np.float32
    assert hand.handedness == "Right"

    detector.close()
