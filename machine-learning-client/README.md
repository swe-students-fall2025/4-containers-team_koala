# Overview

Our system recognizes ASL alphabet hand signs using a lightweight neural network trained on MediaPipe hand landmarks rather than raw images. This approach is fast, efficient, and works consistently across lighting conditions and backgrounds.

We use two core technologies:

- OpenCV: captures frames from the webcam
- Google MediaPipe Hands: detects 21 hand landmarks per frame (each with x, y, z coordinates)

Instead of classifying images directly, the model takes the 63-dimensional landmark vector (21×(x, y, z)), which represents the spatial configuration of the hand.

## Data

Since publicly available ASL datasets do not match our real-world setting, we recorded our own training samples:

- We performed each ASL letter in front of the webcam.
- MediaPipe extracted the (x, y, z) coordinates of the 21 landmarks.
- We normalized each sample by:
  - Translating the wrist landmark to the origin
  - Scaling the hand so its maximum landmark distance equals 1
- We labeled each recording with its corresponding letter.

This gave us a clean and consistent dataset tailored to our environment.

## API Contract

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

## Raw Usage Guide

This guide explains how to:

1. Record training samples  
2. Train the MLP model  
3. Run the real-time ASL demo  
4. Use the trained model in any other Python script or directory  

---

## Install Dependencies

```bash
pipenv install
```

## Record Training Samples

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

## Training the Model

To train classifier on your recorded landmarks run:

```bash
pipenv run python -m src.train_mlp
```

The trained model will be saved to `models/mlp_webcam.pt`

## Demo

To run the model using the demo, use the python command below

```bash
pipenv run python -m src.webcam_demo
```
