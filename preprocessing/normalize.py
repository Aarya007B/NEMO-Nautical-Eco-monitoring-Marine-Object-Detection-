"""
preprocessing/normalize.py — Sonar image normalization.

Side-scan sonar intensity values can span very different ranges across
different sonar systems and missions. Normalization brings pixel values
into a consistent range before feeding into the neural network.

WARNING: Aggressive normalization can destroy sonar-specific information
(e.g., shadow depth relative to target brightness). Use conservatively.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def minmax_normalize(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize image to [0, 1] using min-max scaling.

    Args:
        img: Grayscale image as np.float32 array.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized float32 image in [0, 1].
    """
    lo, hi = img.min(), img.max()
    if hi - lo < eps:
        logger.debug("Image has near-zero dynamic range; returning zeros.")
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo + eps)).astype(np.float32)


def zscore_normalize(
    img: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Z-score normalize the image.

    If mean/std are not provided, they are computed from the image itself.

    Args:
        img: Grayscale image as np.float32 array.
        mean: Optional fixed mean (e.g., dataset-level statistic).
        std:  Optional fixed std  (e.g., dataset-level statistic).
        eps:  Small constant to avoid division by zero.

    Returns:
        Z-score normalized float32 image.
    """
    mu = mean if mean is not None else float(img.mean())
    sigma = std if std is not None else float(img.std())
    if sigma < eps:
        logger.debug("Image standard deviation near zero; returning zeros.")
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mu) / (sigma + eps)).astype(np.float32)


def to_float32(img: np.ndarray) -> np.ndarray:
    """Convert any integer image to float32 scaled to [0, 1]."""
    if img.dtype == np.float32:
        return img
    if img.dtype == np.float64:
        return img.astype(np.float32)
    # Integer types
    max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
    return (img.astype(np.float32) / max_val)
