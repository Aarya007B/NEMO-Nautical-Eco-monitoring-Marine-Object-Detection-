"""
preprocessing/denoise.py — Optional sonar image denoising.

Side-scan sonar speckle noise can reduce detector performance.
However, aggressive denoising can destroy target edge information.
All methods are optional and disabled by default (PRD §19).
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def gaussian_denoise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Gentle Gaussian blur for noise reduction.

    Args:
        img: Float32 grayscale image [H, W].
        kernel_size: Blur kernel size (odd integer).

    Returns:
        Denoised float32 image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (kernel_size, kernel_size), 0)
    return (blurred.astype(np.float32) / 255.0)


def median_denoise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Median filter — good for salt-and-pepper noise in sonar.

    Args:
        img: Float32 grayscale image [H, W].
        kernel_size: Median filter kernel size (odd integer).

    Returns:
        Denoised float32 image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(img_u8, kernel_size)
    return (filtered.astype(np.float32) / 255.0)


def nlm_denoise(
    img: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Non-Local Means denoising — high quality but slow.

    Args:
        img: Float32 grayscale image [H, W].
        h: Filter strength (higher = more smoothing).
        template_window: Template patch size.
        search_window: Search window size.

    Returns:
        Denoised float32 image.
    """
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(
        img_u8,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )
    return (denoised.astype(np.float32) / 255.0)


def denoise(img: np.ndarray, method: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Dispatch to the requested denoising method.

    Args:
        img: Float32 grayscale image.
        method: One of 'gaussian', 'median', 'nlm'.
        **kwargs: Passed through to the selected method.

    Returns:
        Denoised float32 image.
    """
    methods = {
        "gaussian": gaussian_denoise,
        "median": median_denoise,
        "nlm": nlm_denoise,
    }
    if method not in methods:
        raise ValueError(f"Unknown denoise method '{method}'. Choose from: {list(methods)}")
    logger.debug("Denoising with method='%s'", method)
    return methods[method](img, **kwargs)
