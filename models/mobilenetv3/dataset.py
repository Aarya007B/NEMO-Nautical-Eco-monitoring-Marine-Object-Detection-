"""
models/mobilenetv3/dataset.py

CropDataset — dataset for MobileNetV3 verifier training.

Loads sonar candidate crops with binary labels:
    0 = natural
    1 = artificial

Supports:
- Standard binary folders: root/artificial and root/natural
- Marine_PULSE dataset:
    - engineering platform, pipeline or cable -> artificial (1)
    - seabed surface, underwater residual mound -> natural (0)
"""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CropDataset(Dataset):
    """
    Dataset for loading sonar candidate crops for verifier training.
    """

    CLASSES = ["natural", "artificial"]

    # Maps dataset category folder names to binary 0 (natural) or 1 (artificial)
    CLASS_MAPPING: Dict[str, int] = {
        # Standard
        "natural": 0,
        "artificial": 1,
        # Marine_PULSE
        "seabed surface": 0,
        "underwater residual mound": 0,
        "engineering platform": 1,
        "pipeline or cable": 1,
        "pipeline": 1,
        "platform": 1,
        "mound": 0,
    }

    def __init__(
        self,
        root: str,
        crop_size: int = 128,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".bmp"),
    ):
        self.root = Path(root)
        self.crop_size = crop_size
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        if not self.root.exists():
            logger.warning("CropDataset root not found: %s", self.root)
            return

        search_dirs = [self.root]
        if split:
            split_dir = self.root / split
            if split_dir.exists():
                search_dirs = [split_dir]

        # Scan directories
        for base_dir in search_dirs:
            # Check for subdirectories (e.g. train/engineering platform or just engineering platform)
            for sub_dir in base_dir.rglob("*"):
                if sub_dir.is_dir():
                    clean_name = sub_dir.name.lower().strip()
                    if clean_name in self.CLASS_MAPPING:
                        label_idx = self.CLASS_MAPPING[clean_name]
                        for img_path in sorted(sub_dir.iterdir()):
                            if img_path.suffix.lower() in extensions:
                                self.samples.append((str(img_path), label_idx))

        # Fallback: check direct children of root if no recursive match
        if not self.samples:
            for label_idx, cls_name in enumerate(self.CLASSES):
                cls_dir = self.root / cls_name
                if cls_dir.exists():
                    for ext in extensions:
                        for img_path in sorted(cls_dir.glob(f"*{ext}")):
                            self.samples.append((str(img_path), label_idx))

        logger.info(
            "CropDataset: root=%s, split=%s, total_samples=%d (Natural: %d, Artificial: %d)",
            root, split, len(self.samples),
            sum(1 for _, l in self.samples if l == 0),
            sum(1 for _, l in self.samples if l == 1),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Failed to load crop image: %s", path)
            img = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)

        img = cv2.resize(img, (self.crop_size, self.crop_size))
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label
