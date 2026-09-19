"""
preprocessing/resize.py — Letterbox resize for sonar images.

Letterbox padding preserves the original aspect ratio and pads with a
constant value (sonar background mean) rather than stretching the image.
This prevents the detector from seeing distorted sonar geometry.
"""
import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    pad_value: float = 0.0,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target_size x target_size with letterbox padding.

    Args:
        img: Float32 grayscale image [H, W] in [0, 1].
        target_size: Target square dimension (must be divisible by stride).
        pad_value: Constant fill value for padding (default 0).
        stride: Network stride; target_size must be divisible by this.

    Returns:
        (padded_img, scale, (pad_top, pad_left))
        - padded_img: [target_size, target_size] float32 array.
        - scale: Scaling factor applied to original image.
        - (pad_top, pad_left): Padding offsets for coordinate recovery.
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # Resize with INTER_LINEAR (good balance of quality/speed)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute symmetric padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top  = pad_h // 2
    pad_left = pad_w // 2

    # Apply padding
    padded = np.full((target_size, target_size), pad_value, dtype=np.float32)
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    logger.debug(
        "Letterbox: (%d,%d) -> (%d,%d), scale=%.4f, pad=(%d,%d)",
        h, w, target_size, target_size, scale, pad_top, pad_left,
    )
    return padded, scale, (pad_top, pad_left)


def unletterbox_bbox(
    x1: float, y1: float, x2: float, y2: float,
    scale: float,
    pad_top: int,
    pad_left: int,
) -> Tuple[float, float, float, float]:
    """
    Invert letterbox transform for a bounding box.

    Converts detector-space (padded image) coordinates back to
    original image pixel coordinates.

    Args:
        x1, y1, x2, y2: Bbox in padded image space.
        scale: Scale factor used during letterbox.
        pad_top, pad_left: Padding offsets used during letterbox.

    Returns:
        (x1, y1, x2, y2) in original image space.
    """
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top)  / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top)  / scale
    return x1, y1, x2, y2
