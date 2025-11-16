from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.mediapipe_utils import (
    MediaPipeHandDetector,
    normalize_landmarks,
    draw_hand_landmarks_on_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"
OUT_PATH = DATA_DIR / "webcam_landmarks.npz" 


def load_letter_to_index() -> Dict[str, int]:
    with open(LABEL_MAP_PATH, "r") as f:
        mapping = json.load(f)
    return {k: int(v) for k, v in mapping["letter_to_index"].items()}


def main():
    letter_to_index = load_letter_to_index()

    print("Loaded label map with letters:", sorted(letter_to_index.keys()))

    cap = cv2.VideoCapture(0)
    detector = MediaPipeHandDetector(max_num_hands=1, detection_confidence=0.7)

    X: List[np.ndarray] = []
    y: List[int] = []

    print("Press a letter key (A, B, C, ...) to record a sample for that class.")
    print("Press 'q' to quit and save.")

    current_letter = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detect hand landmarks
        hands = detector.process(frame)

        if hands:
            draw_hand_landmarks_on_frame(frame, hands[0])

        # Draw a simple hint
        if current_letter is not None:
            cv2.putText(
                frame,
                f"Recording label: {current_letter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Press letter key to set label; 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Record ASL samples", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting and saving...")
            break

        # If the key is a letter, update current label
        if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
            letter = chr(key).upper()
            if letter in letter_to_index:
                current_letter = letter
                print(f"Current label set to: {letter}")
            else:
                print(f"Letter '{letter}' not in label_map.json; ignoring.")
            continue

        # If we have a current label AND a hand is detected, record a sample
        if current_letter is not None and hands:
            hl = hands[0]
            feat = normalize_landmarks(hl.points)
            X.append(feat)
            y.append(letter_to_index[current_letter])

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    X = np.stack(X, axis=0) if X else np.zeros((0, 63), dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, X=X, y=y)

    print(f"Saved {len(y)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
