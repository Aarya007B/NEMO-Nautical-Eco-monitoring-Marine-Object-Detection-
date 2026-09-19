"""
scoring/features.py — Acoustic feature extraction from sonar crops.

Extracts sonar-domain evidence features that complement neural network
predictions. These features are used by EvidenceFusion alongside model
probabilities.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from inference.types import AcousticFeatures, BoundingBox

logger = logging.getLogger(__name__)


class AcousticFeatureExtractor:
    """
    Extracts acoustic evidence features from a sonar frame and detection bbox.

    All features are normalized to [0, 1] for consistent fusion weighting.
    """

    def extract(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> AcousticFeatures:
        """
        Extract acoustic features for a single detection.

        Args:
            frame: Grayscale sonar frame [H, W] as float32 in [0, 1].
            bbox: Detection bounding box in pixel coordinates.

        Returns:
            AcousticFeatures with all fields populated.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid bbox for feature extraction: %s", bbox)
            return AcousticFeatures()

        crop = frame[y1:y2, x1:x2]

        return AcousticFeatures(
            local_contrast=self._local_contrast(crop),
            intensity_difference=self._intensity_difference(frame, crop, x1, y1, x2, y2),
            bounding_box_area=self._normalized_area(bbox, h, w),
            aspect_ratio=bbox.aspect_ratio,
            edge_density=self._edge_density(crop),
            image_quality=self._image_quality(frame),
        )

    def _local_contrast(self, crop: np.ndarray) -> float:
        """Normalized standard deviation of the crop."""
        std = float(crop.std())
        return min(std / 0.3, 1.0)  # 0.3 is a high-contrast reference

    def _intensity_difference(
        self, frame: np.ndarray, crop: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> float:
        """Absolute intensity difference between crop and surrounding region."""
        h, w = frame.shape[:2]
        pad = max((x2 - x1), (y2 - y1))
        sx1 = max(0, x1 - pad)
        sy1 = max(0, y1 - pad)
        sx2 = min(w, x2 + pad)
        sy2 = min(h, y2 + pad)
        surround = frame[sy1:sy2, sx1:sx2]
        diff = abs(float(crop.mean()) - float(surround.mean()))
        return min(diff / 0.3, 1.0)

    def _normalized_area(self, bbox: BoundingBox, h: int, w: int) -> float:
        """Bbox area normalized by frame area."""
        frame_area = max(h * w, 1)
        return min(bbox.area / frame_area, 1.0)

    def _edge_density(self, crop: np.ndarray) -> float:
        """Canny edge pixel density in the crop."""
        crop_u8 = (crop * 255).clip(0, 255).astype(np.uint8)
        edges = cv2.Canny(crop_u8, 50, 150)
        total_pixels = max(edges.size, 1)
        return float(edges.sum() / 255) / total_pixels

    def _image_quality(self, frame: np.ndarray) -> float:
        """Simple Laplacian variance-based quality estimate."""
        frame_u8 = (frame * 255).clip(0, 255).astype(np.uint8)
        lap_var = cv2.Laplacian(frame_u8, cv2.CV_64F).var()
        return min(lap_var / 500.0, 1.0)
