from pathlib import Path
import json
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image


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

    with open(label_map_path, "r") as f:
        mapping = json.load(f)

    index_to_letter = {int(k): v for k, v in mapping["index_to_letter"].items()}
    letter_to_index = {k: int(v) for k, v in mapping["letter_to_index"].items()}
    raw_label_to_index = {
        int(k): int(v) for k, v in mapping["raw_label_to_index"].items()
    }
    return index_to_letter, letter_to_index, raw_label_to_index


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
        self.dataset = load_dataset("Voxel51/American-Sign-Language-MNIST", split=split)

        (
            self.index_to_letter,
            self.letter_to_index,
            self.raw_label_to_index,
        ) = load_label_maps()

        self.transform = transform
        self.as_pil = as_pil

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # Convert to PIL for consistency
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        raw_label = int(sample["label"])
        label = self.raw_label_to_index[raw_label]

        if self.as_pil:
            return img, label

        # Convert to tensor if no custom transform is provided
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # Minimal manual conversion to 1x28x28 float tensor in [0, 1]
            img_tensor = (
                torch.from_numpy(
                    (
                        torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                        .view(img.size[1], img.size[0])
                        .numpy()
                    )
                )
                .unsqueeze(0)
                .float()
                / 255.0
            )

        return img_tensor, label
