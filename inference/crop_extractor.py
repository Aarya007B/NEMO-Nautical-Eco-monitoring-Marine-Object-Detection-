"""
inference/crop_extractor.py — Crop extraction from sonar frames.

Extracts and saves detection crops for the Stage-2 verifier.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from inference.types import BoundingBox

logger = logging.getLogger(__name__)


class CropExtractor:
    """
    Extracts crops from sonar frames for the MobileNetV3 verifier.

    Args:
        crop_size: Target crop size (square).
        bbox_padding: Fractional padding around bounding box.
        save_dir: Optional directory to save crops.
    """

    def __init__(
        self,
        crop_size: int = 128,
        bbox_padding: float = 0.15,
        save_dir: Optional[str] = None,
    ):
        self.crop_size = crop_size
        self.bbox_padding = bbox_padding
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        detection_id: str = "",
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """
        Extract a crop from the frame at the given bounding box.

        Args:
            frame: Grayscale sonar frame [H, W] as float32 in [0, 1].
            bbox: Detection bounding box in pixel coordinates.
            detection_id: Unique ID for saving the crop.

        Returns:
            Tuple of (crop_tensor [1, 1, crop_size, crop_size], saved_path).
        """
        h, w = frame.shape[:2]

        # Expand bbox with padding
        expanded = bbox.expand(self.bbox_padding, w, h)

        # Extract crop region
        x1 = max(0, int(expanded.x1))
        y1 = max(0, int(expanded.y1))
        x2 = min(w, int(expanded.x2))
        y2 = min(h, int(expanded.y2))

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid crop region for %s: [%d,%d,%d,%d]", detection_id, x1, y1, x2, y2)
            crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        else:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        # Save crop if directory configured
        saved_path = None
        if self.save_dir and detection_id:
            saved_path = str(self.save_dir / f"{detection_id}.png")
            crop_u8 = (crop * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(saved_path, crop_u8)

        # Convert to tensor [1, 1, H, W]
        tensor = torch.from_numpy(crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        return tensor, saved_path
