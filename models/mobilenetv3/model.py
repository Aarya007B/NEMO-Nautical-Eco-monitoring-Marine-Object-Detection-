"""
models/mobilenetv3/model.py

MobileNetV3-Small — Second-stage sonar candidate verifier.

PURPOSE:
    The detector (Stage 1) answers: "Does this region look like a candidate target?"
    This verifier (Stage 2) answers: "Does this candidate look more like an
    artificial/man-made object than a natural seabed feature?"

    The two outputs are stored INDEPENDENTLY (PRD §13):
        detector_confidence
        verifier_artificial_probability

NEMO ADAPTATION:
    Input: single-channel sonar crop [B, 1, 128, 128]
    (torchvision MobileNetV3 normally expects 3-channel RGB)

OUTPUT:
    VerificationResult with:
        natural_probability
        artificial_probability
        predicted_class  ("natural" or "artificial")
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.types import VerificationResult

logger = logging.getLogger(__name__)

CLASS_NAMES = ["natural", "artificial"]


class MobileNetV3Verifier(nn.Module):
    """
    MobileNetV3-Small binary classifier for sonar candidate verification.

    Classifies sonar crops as:
        0 = natural  (rocks, ridges, coral, sand, shadows, artifacts)
        1 = artificial (man-made debris, ghost nets, UXO, pipelines, etc.)

    Args:
        in_channels (int):          Input channels (1 for sonar).
        num_classes (int):          Output classes (2: natural/artificial).
        pretrained (bool):          Attempt to load ImageNet weights.
        conf_threshold (float):     Threshold for 'artificial' prediction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        pretrained: bool = False,
        conf_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold

        # Load torchvision MobileNetV3-Small
        try:
            import torchvision.models as tvm
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = tvm.mobilenet_v3_small(weights=weights)
        except Exception as e:
            raise ImportError(
                f"torchvision is required for MobileNetV3Verifier: {e}"
            ) from e

        # Adapt first convolution for 1-channel sonar input
        # SOURCE: NEMO adaptation (NOT part of original MobileNetV3)
        if in_channels != 3:
            orig_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            if pretrained and orig_conv.weight is not None:
                new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            backbone.features[0][0] = new_conv
            logger.info(
                "MobileNetV3Verifier: adapted first conv from 3-ch to %d-ch",
                in_channels,
            )

        # Adapt classifier head for num_classes
        in_feats = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_feats, num_classes)

        self.backbone = backbone

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MobileNetV3Verifier: in_ch=%d, classes=%d, params=%.2fM",
            in_channels, num_classes, total_params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward pass. Returns [B, 2] logits."""
        return self.backbone(x)

    @torch.no_grad()
    def verify(self, crop: torch.Tensor) -> VerificationResult:
        """
        Run verifier on a single sonar crop.

        Args:
            crop: [1, 1, H, W] sonar crop tensor.

        Returns:
            VerificationResult with natural/artificial probabilities.
        """
        was_training = self.training
        self.eval()

        logits = self.forward(crop)
        probs = F.softmax(logits, dim=-1)[0]

        natural_prob = probs[0].item()
        artificial_prob = probs[1].item()
        pred_class = (
            "artificial" if artificial_prob >= self.conf_threshold else "natural"
        )

        result = VerificationResult(
            natural_probability=natural_prob,
            artificial_probability=artificial_prob,
            predicted_class=pred_class,
            raw_logits=logits[0].tolist(),
        )

        if was_training:
            self.train()

        return result

    def load_weights(self, weights_path: str) -> None:
        """Load verifier weights from a .pt file."""
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Verifier weights not found: {weights_path}\n"
                "Place weights in models/mobilenetv3/weights/. See README.md."
            )
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=False)
        logger.info("Loaded verifier weights from: %s", weights_path)
