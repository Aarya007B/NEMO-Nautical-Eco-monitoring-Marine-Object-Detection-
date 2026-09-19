"""
preprocessing/contrast.py — Optional sonar contrast enhancement.

Side-scan sonar targets can have very low contrast against the seabed.
CLAHE is the preferred method for sonar as it preserves local structure.

"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def clahe_enhance(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Recommended for sonar imagery: enhances local contrast while limiting
    noise amplification in smooth regions (seabed, water column).

    Args:
        img: Float32 grayscale image [H, W] in [0, 1].
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for local histogram computation.

    Returns:
        Contrast-enhanced float32 image.
    """
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_u8)
    return (enhanced.astype(np.float32) / 255.0)


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization.

    Less preferred for sonar than CLAHE; can over-amplify background noise.

    Args:
        img: Float32 grayscale image [H, W] in [0, 1].

    Returns:
        Equalized float32 image.
    """
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_u8)
    return (equalized.astype(np.float32) / 255.0)


def enhance_contrast(
    img: np.ndarray,
    method: str = "clahe",
    **kwargs,
) -> np.ndarray:
    """
    Dispatch to the requested contrast enhancement method.

    Args:
        img: Float32 grayscale image in [0, 1].
        method: One of 'clahe', 'histogram_eq'.
        **kwargs: Passed to the selected method.

    Returns:
        Contrast-enhanced float32 image.
    """
    methods = {
        "clahe": clahe_enhance,
        "histogram_eq": histogram_equalize,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown contrast method '{method}'. Choose from: {list(methods)}"
        )
    logger.debug("Contrast enhancement with method='%s'", method)
    return methods[method](img, **kwargs)
