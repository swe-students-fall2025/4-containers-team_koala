"""
Module that handles ASL Mnist Dataset. NOT CURRENTLY USED IN MODEL
"""

from pathlib import Path

import json
from typing import Optional, Callable, Tuple
import time

from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np
import mediapipe as mp



def load_label_maps(label_map_path: Optional[Path] = None):
    """
    Load index/label mappings from data/label_map.json.

    Returns:
        index_to_letter (dict[int, str])
        letter_to_index (dict[str, int])
        raw_label_to_index (dict[int, int])
    """
    if label_map_path is None:
        label_map_path = Path(__file__).resolve().parents[1] / "data" / "label_map.json"

    with open(label_map_path, "r", encoding='utf8') as f:
        mapping = json.load(f)

    index_to_letter = {int(k): v for k, v in mapping["index_to_letter"].items()}
    letter_to_index = {k: int(v) for k, v in mapping["letter_to_index"].items()}
    raw_label_to_index = {
        int(k): int(v) for k, v in mapping["raw_label_to_index"].items()
    }
    return index_to_letter, letter_to_index, raw_label_to_index


def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks:
    - Translate so wrist is at origin
    - Scale so max distance = 1
    """
    assert pts.shape == (21, 3)

    wrist = pts[0].copy()
    centered = pts - wrist

    dists = np.linalg.norm(centered, axis=1)
    max_dist = np.max(dists)

    if max_dist > 0:
        centered /= max_dist

    return centered.flatten().astype(np.float32)


mp_hands = mp.solutions.hands
_mp_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


def load_asl_mnist_with_retries(split="train", max_retries=1000, base_delay=60):
    """
    Loads ASL MNIST dataset with exponenetial backoffs to handle throttling from HF

    Args:
        split: ['train', 'test', 'validation']. Indicates type of split for data. Defaults to train
        max_retries: int - indicates how many retries
        base_delay: int - The starting delay for the process to wait before retrying

    Returns:
        ASL MNIST HF dataset
    """
    for attempt in range(max_retries):
        try:
            return load_dataset("Voxel51/American-Sign-Language-MNIST", split=split)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait = base_delay * (2**attempt)
                print(
                    f"Got 429 from HF (attempt {attempt + 1}/{max_retries}), sleeping {wait}s..."
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries while loading ASL MNIST")


class ASLMNISTDataset(Dataset):
    """
    Wrapper around Hugging Face Voxel51/American-Sign-Language-MNIST.

    - Downloads/loads the dataset via datasets.load_dataset
    - Converts HF's non-contiguous labels to contiguous indices
      using raw_label_to_index from label_map.json
    - Returns (image_tensor, label_index)
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        as_pil: bool = False,
    ):
        """
        Args:
            split: "train", "test", or "validation"
            transform: Optional torchvision-style transform for images.
                       If None, images will be converted to float32
                       tensors in [0, 1] with shape (1, 28, 28).
            as_pil: If True, __getitem__ returns PIL.Image instead of tensor
        """
        self.dataset = load_asl_mnist_with_retries(split=split)

        (
            self.index_to_letter,
            self.letter_to_index,
            self.raw_label_to_index,
        ) = load_label_maps()

        self.transform = transform
        self.as_pil = as_pil

    def __len__(self):
        return len(self.dataset)

    def _extract_landmarks(self, pil_img: Image.Image) -> Optional[np.ndarray]:
        """Runs MediaPipe on a PIL image, returns (21,3) or None."""
        np_img = np.array(pil_img)
        results = _mp_model.process(np_img)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        return pts

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        sample = self.dataset[idx]

        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        raw_label = int(sample["label"])
        label = self.raw_label_to_index[raw_label]

        pts = self._extract_landmarks(img)

        if pts is None:
            return np.zeros((63,), dtype=np.float32), label

        X = normalize_landmarks(pts)  # shape (63,)
        return X, label
