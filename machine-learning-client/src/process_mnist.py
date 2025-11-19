"""
Script for processing MNIST dataset. Downloads to asl_mnist_landmarks.npz in data dir
"""

from pathlib import Path

import numpy as np
from .dataset_asl_mnist import ASLMNISTDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = DATA_DIR / "asl_mnist_landmarks.npz"

ds_train = ASLMNISTDataset(split="train")

X = []
y = []

for xi, yi in enumerate(ds_train):
    X.append(xi)
    y.append(yi)

np.savez(OUT_PATH, X=X, y=y)
