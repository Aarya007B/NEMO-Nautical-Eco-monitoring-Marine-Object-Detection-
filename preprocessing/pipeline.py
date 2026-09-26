"""
preprocessing/pipeline.py — SonarPreprocessor orchestrates all steps.

Pipeline:
    Raw image path / numpy array
          ↓
    Load & grayscale (single-channel)
          ↓
    Convert to float32 [0, 1]
          ↓
    Normalization
          ↓
    Optional denoising
          ↓
    Optional contrast enhancement
          ↓
    Letterbox resize + padding
          ↓
    Tensor [1, H, W]  (NEMO 1-channel format)
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from preprocessing.normalize import minmax_normalize, to_float32
from preprocessing.denoise import denoise
from preprocessing.contrast import enhance_contrast
from preprocessing.resize import letterbox

logger = logging.getLogger(__name__)


class SonarPreprocessor:
    """
    Configurable sonar image preprocessing pipeline.

    Converts raw sonar images (any supported format) into a
    1-channel [1, H, W] float32 PyTorch tensor ready for the
    YOLO11n-1C detector (validated MVP).

    The validated MVP default uses raw native input (normalize=false).
    Normalization, denoising, and contrast enhancement are available
    as configurable options but were not selected based on T1/T3
    ablation experiments.

    The raw image is NEVER modified on disk.

    Args:
        config: Preprocessing config dict (from config.yaml[preprocessing]).
    """

    def __init__(self, config: Dict):
        self.image_size    = int(config.get("image_size", 640))
        self.do_normalize  = bool(config.get("normalize", True))

        denoise_cfg        = config.get("denoise", {})
        self.do_denoise    = bool(denoise_cfg.get("enabled", False))
        self.denoise_method = denoise_cfg.get("method", "gaussian")
        self.denoise_ksize = int(denoise_cfg.get("kernel_size", 3))

        contrast_cfg       = config.get("contrast", {})
        self.do_contrast   = bool(contrast_cfg.get("enabled", False))
        self.contrast_method = contrast_cfg.get("method", "clahe")
        self.clip_limit    = float(contrast_cfg.get("clip_limit", 2.0))
        self.tile_grid     = tuple(contrast_cfg.get("tile_grid_size", [8, 8]))

        logger.info(
            "SonarPreprocessor: size=%d normalize=%s denoise=%s contrast=%s",
            self.image_size, self.do_normalize, self.do_denoise, self.do_contrast,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """
        Preprocess a sonar image.

        Args:
            source: File path (str/Path) or numpy array [H, W] or [H, W, C].

        Returns:
            (tensor, scale, (pad_top, pad_left))
            - tensor: [1, image_size, image_size] float32 torch.Tensor
            - scale: letterbox scale factor (for bbox recovery)
            - (pad_top, pad_left): letterbox padding (for bbox recovery)
        """
        img = self._load(source)
        img = self._to_single_channel(img)
        img = to_float32(img)

        if self.do_normalize:
            img = minmax_normalize(img)

        if self.do_denoise:
            img = denoise(
                img,
                method=self.denoise_method,
                kernel_size=self.denoise_ksize,
            )

        if self.do_contrast:
            img = enhance_contrast(
                img,
                method=self.contrast_method,
                clip_limit=self.clip_limit,
                tile_grid_size=self.tile_grid,
            )

        img, scale, padding = letterbox(img, target_size=self.image_size)

        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        return tensor, scale, padding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or return array as-is."""
        if isinstance(source, np.ndarray):
            return source
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        return img

    def _to_single_channel(self, img: np.ndarray) -> np.ndarray:
        """
        Convert any image to single-channel.

        For sonar images that arrive as RGB/BGR (e.g., PNG saved by third-party
        software), we average channels rather than discarding information.
        The internal representation is always [H, W] float32.
        """
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0]
            # BGR or RGB — compute luminance-weighted mean
            # Not using cv2.COLOR_BGR2GRAY to avoid implicit assumptions
            # about channel ordering in third-party sonar exports.
            return img.mean(axis=2).astype(img.dtype)
        raise ValueError(f"Unexpected image shape: {img.shape}")
