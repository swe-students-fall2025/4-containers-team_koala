from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.model_MLP import LandmarkMLP

from src.mediapipe_utils import (
    MediaPipeHandDetector,
    normalize_landmarks,
    draw_hand_landmarks_on_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"
MODEL_PATH = MODELS_DIR / "mlp_webcam.pt"


def load_index_to_letter():
    """
    Loads index to letter mappings from data/label_map.json file
    """
    with open(LABEL_MAP_PATH, "r") as f:
        mapping = json.load(f)
    index_to_letter = {int(k): v for k, v in mapping["index_to_letter"].items()}
    return index_to_letter


def main():
    index_to_letter = load_index_to_letter()
    num_classes = len(index_to_letter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkMLP(input_dim=63, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    detector = MediaPipeHandDetector(max_num_hands=1)
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        hands = detector.process(frame)
        pred_letter = "-"
        pred_conf = 0.0

        # when we detect a hand, draw landmarks and predict letter only if confidence > 0.5
        if hands:
            draw_hand_landmarks_on_frame(frame, hands[0])

            feat = normalize_landmarks(hands[0].points)
            x = torch.from_numpy(feat).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                conf, idx = probs.max(dim=1)
                conf = conf.item()
                idx = idx.item()

            if conf > 0.5:
                pred_letter = index_to_letter.get(idx, "?")
                pred_conf = conf
            else:
                pred_letter = "-"
                pred_conf = conf

        text = f"Pred: {pred_letter} ({pred_conf:.2f})"
        cv2.putText(
            frame,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ASL Alphabet Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
