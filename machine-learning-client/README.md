# API Contract

## POST
Requests from the frontend should be the raw MediaPipe landmarks. No changes done to the data
```bash
POST /predict
{
  "points": [[x0, y0, z0], ..., [x20, y20, z20]]
}
```

### Response
```bash
{
  "letter": "S",
  "confidence": 0.97
}
```
The response returned will be a JSON object containing the letter and the prediction confidence.


# Raw Usage Guide

This guide explains how to:

1. Record training samples  
2. Train the MLP model  
3. Run the real-time ASL demo  
4. Use the trained model in any other Python script or directory  

---

# Install Dependencies

```bash
pipenv install
```

# Record Training Samples

You can collect webcam data so the model learns your handshape, camera angle, lighting, etc.

Run:

```bash
pipenv run python -m src.record_webcam_samples
```

This will create the output data file `data/webcam_landmarks.npz` which is later used to train the model

## Controls

| Key | Action                           |
| --- | -------------------------------- |
| A–Z | Set current label to that letter |
| q   | Quit and save dataset            |


# Training the Model

To train classifier on your recorded landmarks run:

```bash
pipenv run python -m src.train_mlp
```

The trained model will be saved to `models/mlp_webcam.pt`

# Demo

To run the model using the demo, use the python command below

```bash
pipenv run python -m src.webcam_demo
```

