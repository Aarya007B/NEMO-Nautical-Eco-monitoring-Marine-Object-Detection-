"""
inference/verifier.py — CandidateVerifier wrapper.

Wraps the MobileNetV3 model with crop extraction for the pipeline.
"""
import logging
from typing import Dict, Optional

import torch

from inference.types import VerificationResult
from models.mobilenetv3.model import MobileNetV3Verifier

logger = logging.getLogger(__name__)


class CandidateVerifier:
    """
    Stage-2 candidate verifier.

    Wraps MobileNetV3Verifier for use in the inference pipeline.

    Args:
        verifier_config: Verifier config dict from config.yaml.
        device: Target device.
    """

    def __init__(self, verifier_config: Dict, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.conf_threshold = verifier_config.get("confidence_threshold", 0.50)

        self.model = MobileNetV3Verifier(
            in_channels=1,
            num_classes=2,
            conf_threshold=self.conf_threshold,
        ).to(self.device)
        self.model.eval()

        # Attempt to load weights
        weights_path = verifier_config.get("weights", "")
        if weights_path:
            try:
                self.model.load_weights(weights_path)
                logger.info("Loaded verifier weights: %s", weights_path)
            except FileNotFoundError:
                logger.warning(
                    "Verifier weights not found at %s. "
                    "Running with random weights (smoke test mode).",
                    weights_path,
                )

        logger.info("CandidateVerifier: device=%s", self.device)

    @torch.no_grad()
    def verify(self, crop: torch.Tensor) -> VerificationResult:
        """
        Verify a single crop.

        Args:
            crop: [1, 1, H, W] crop tensor.

        Returns:
            VerificationResult.
        """
        crop = crop.to(self.device)
        return self.model.verify(crop)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
