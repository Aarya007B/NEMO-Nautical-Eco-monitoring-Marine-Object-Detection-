"""
models/rcdi_yolo/dataset.py

SonarDetectionDataset — dataset for RCDI-YOLO training and validation.

Loads sonar images and their YOLO-format annotations.

Directory structure:
    images/
        image_001.png
        image_002.png
    labels/
        image_001.txt
        image_002.txt

YOLO annotation format per line:
    class cx cy w h   (normalized 0-1)
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SonarDetectionDataset(Dataset):
    """
    Dataset for RCDI-YOLO detection training.

    Args:
        image_paths: List of image file paths.
        annotation_paths: List of annotation file paths (same length).
        image_size: Input image size (default 640).
        num_classes: Number of detection classes.
        augment: Whether to apply augmentation (for training).
    """

    def __init__(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        image_size: int = 640,
        num_classes: int = 1,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment

        # Filter to only existing image paths
        self.samples = []
        for img_path, ann_path in zip(image_paths, annotation_paths):
            if Path(img_path).exists():
                self.samples.append((img_path, ann_path))

        logger.info(
            "SonarDetectionDataset: %d images, image_size=%d, num_classes=%d",
            len(self.samples), image_size, num_classes,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path, ann_path = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Load annotations
        targets = self._load_annotations(ann_path, img.shape[:2])

        # Resize
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        img_float = img_resized.astype(np.float32) / 255.0

        # Convert to tensor [1, H, W]
        tensor = torch.from_numpy(img_float).unsqueeze(0)

        if self.augment:
            tensor = self._augment(tensor)

        return tensor, targets

    def _load_annotations(self, ann_path: str, img_shape: Tuple[int, int]) -> torch.Tensor:
        """Load YOLO-format annotations."""
        if not ann_path or not Path(ann_path).exists():
            return torch.zeros((0, 6))

        targets = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = [float(p) for p in parts[1:5]]
                    targets.append([cls_id, cx, cy, bw, bh])

        if not targets:
            return torch.zeros((0, 6))

        return torch.tensor(targets, dtype=torch.float32)

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simple augmentation: horizontal flip and intensity jitter."""
        if torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[2])
        jitter = torch.randn(1).item() * 0.05
        tensor = torch.clamp(tensor + jitter, 0.0, 1.0)
        return tensor

    @property
    def num_samples(self) -> int:
        return len(self.samples)
