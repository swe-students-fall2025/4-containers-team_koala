"""
API for ML client; handles client interactions with model
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any
import logging

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS

from models.model_MLP import LandmarkMLP
from .mediapipe_utils import normalize_landmarks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_MAP_PATH = DATA_DIR / "label_map.json"
MODEL_PATH = MODELS_DIR / "mlp_webcam.pt"

with LABEL_MAP_PATH.open("r") as f:
    label_map = json.load(f)

INDEX_TO_LETTER = {int(k): v for k, v in label_map["index_to_letter"].items()}
NUM_CLASSES = len(INDEX_TO_LETTER)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LandmarkMLP(input_dim=63, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loaded model from %s", MODEL_PATH)

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health() -> Any:
    """
    Returns status code 200 indicating healthy if reachable
    """
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """
    Predict an ASL letter from raw MediaPipe hand landmarks.

    Expected JSON body:
    {
      "points": [
        [x0, y0, z0],
        [x1, y1, z1],
        ...
        [x20, y20, z20]
      ]
    }
    where:
      - length of points must be 21
      - each inner list must have length 3
      - this is the raw MediaPipe format

    Returns:
    {
      "letter": [str]
      "confidence": [int]
    }
    """

    # Data validation
    data = request.get_json(silent=True)
    if data is None:
        logger.error("ERROR: Empty request")
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    points = data.get("points")
    if points is None:
        logger.error("ERROR: No 'points' filed in request: %s", data)
        return jsonify({"error": "Missing 'points' field in request body"}), 400

    if not isinstance(points, list) or len(points) != 21:
        logger.error("ERROR: Points is not expected shape: %s", data)
        return (
            jsonify(
                {"error": "Expected 'points' to be a list of length 21 (21 landmarks)"}
            ),
            400,
        )

    try:
        pts_array = np.asarray(points, dtype=np.float32)
    except ValueError as e:
        return (
            jsonify({"error": f"Could not convert 'points' to a float32 array {e}"}),
            400,
        )

    if pts_array.shape != (21, 3):
        return (
            jsonify(
                {
                    "error": (
                        "Expected 'points' shape (21, 3). "
                        f"Got {list(pts_array.shape)} instead."
                    )
                }
            ),
            400,
        )

    # Data is ok -> normalize and predict
    feats = normalize_landmarks(pts_array)
    x = torch.from_numpy(feats).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        confidence = float(conf.item())
        pred_idx = int(idx.item())

    letter = INDEX_TO_LETTER.get(pred_idx, "?")

    return (
        jsonify(
            {
                "letter": letter,
                "confidence": confidence,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
