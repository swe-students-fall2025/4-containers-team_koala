from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS


@dataclass
class HandLandmarks:
    """
    Represents a single hand's 21 landmarks as a (21, 3) numpy array
    in normalized image coordinates [0,1], plus handedness info.
    """
    points: np.ndarray
    handedness: str


class MediaPipeHandDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """
        Run MediaPipe Hands on a BGR frame.
        Returns a list of HandLandmarks objects (possibly empty).
        """
        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        landmarks_list: List[HandLandmarks] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                pts = []
                for lm in hand_landmarks.landmark:
                    pts.append([lm.x, lm.y, lm.z])
                pts = np.array(pts, dtype=np.float32)
                label = handedness.classification[0].label
                landmarks_list.append(HandLandmarks(points=pts, handedness=label))

        return landmarks_list

    def close(self):
        self.hands.close()


def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks:
    - Translate so wrist (index 0) is at origin.
    - Scale so that the max distance to any point is 1.
    Returns a flattened vector of shape (63,) for (21,3).

    Args:
        pts: np.ndarray of shape (21, 3) in normalized image coords.

    Returns:
        np.ndarray of shape (63,) float32.
    """
    assert pts.shape == (21, 3)
    wrist = pts[0].copy()
    centered = pts - wrist 

    # Use max L2 distance to any point
    dists = np.linalg.norm(centered, axis=1)
    max_dist = np.max(dists)
    if max_dist > 0:
        centered /= max_dist

    return centered.flatten().astype(np.float32)

def draw_hand_landmarks_on_frame(frame_bgr: np.ndarray, hand: HandLandmarks) -> None:
    """
    Draw simple circles and connection lines for a single hand's landmarks
    onto the given BGR frame (in-place).

    Args:
        frame_bgr: np.ndarray of shape (H, W, 3), OpenCV BGR image.
        hand: HandLandmarks with points in normalized [0,1] coordinates.
    """
    h, w, _ = frame_bgr.shape
    pts = hand.points 

    # Convert normalized (x,y) to pixel coords
    pixel_pts = []
    for x, y, z in pts:
        px = int(x * w)
        py = int(y * h)
        pixel_pts.append((px, py))

    # Draw connections
    for connection in HAND_CONNECTIONS:
        i = connection[0]
        j = connection[1]
        x1, y1 = pixel_pts[i]
        x2, y2 = pixel_pts[j]
        cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmark points
    for (px, py) in pixel_pts:
        cv2.circle(frame_bgr, (px, py), 4, (0, 0, 255), -1)

    cv2.putText(
        frame_bgr,
        hand.handedness,
        (pixel_pts[0][0] + 5, pixel_pts[0][1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
