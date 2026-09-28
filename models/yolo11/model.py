"""
models/yolo11/model.py — NEMO Validated MVP Detector (YOLO11n-1C).

=============================================================================
NEMO ARCHITECTURE DEFINITION: YOLO11n-1C (CASE C)
=============================================================================

The validated NEMO detector is YOLO11n with a NATIVE 1-CHANNEL INPUT LAYER.
The model itself was adapted to accept a single-channel tensor [B, 1, 640, 640].

The first convolution was explicitly adapted from:
    Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
to:
    Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

WEIGHT INITIALIZATION:
    For the original pretrained first convolution with weights:
        W ∈ R^(C_out × 3 × k × k)
    the native 1-channel weights were initialized by:
        W_new = mean(W, dim=channel)
    which produces:
        W_new ∈ R^(C_out × 1 × k × k)

    The native 1-channel first-layer weights were initialized from the original
    pretrained 3-channel filters by averaging across the input-channel dimension,
    while the remaining compatible pretrained layers were retained.

CONFIGURATIONS COMPARISON:
    CASE A — Original 3-channel YOLO:
        Input: [B, 3, H, W]
        Model: Conv2d(3, 16, ...)
        Optical RGB model.

    CASE B — Grayscale replicated to 3 channels:
        Input: [I, I, I]
        Model: Conv2d(3, 16, ...)
        3-channel architecture with replicated data.

    CASE C — NEMO YOLO11n-1C (Validated Architecture):
        Input: [B, 1, H, W]
        Model: Conv2d(1, 16, ...)
        Native 1-channel architecture (T1-B validated MVP).

DATA FLOW:
    Raw SSS image
          ↓
    Native single-channel image
          ↓
    640 × 640 resizing / letterboxing
          ↓
    Tensor [B, 1, 640, 640]
          ↓
    YOLO11n-1C (Conv2d(1, 16, 3, 3))
          ↓
    Bounding boxes + class + detector confidence
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics.utils import nms

from inference.types import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YOLO11nDetector(nn.Module):
    """
    YOLO11n-1C detector for native 1-channel sonar images.

    Executes inference directly on native single-channel acoustic tensors
    [B, 1, 640, 640] using an adapted Conv2d(1, 16, 3, 3) first convolution layer.

    Args:
        weights_path:     Path to .pt weights file.
        num_classes:      Number of detection classes (default 4).
        conf_threshold:   Confidence threshold for filtering.
        nms_iou:          NMS IoU threshold.
        image_size:       Expected input size.
        in_channels:      Input channels (1 for native sonar).
        demo_mode:        If True, allow running without weights (empty output).
    """

    CLASS_NAMES = {
        0: "aircraft",
        1: "fish",
        2: "other",
        3: "shipwreck",
    }

    def __init__(
        self,
        weights_path: str = "models/yolo11/weights/yolo11n_1c.pt",
        num_classes: int = 4,
        conf_threshold: float = 0.25,
        nms_iou: float = 0.70,
        image_size: int = 640,
        in_channels: int = 1,
        demo_mode: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.image_size = image_size
        self.in_channels = in_channels
        self.demo_mode = demo_mode
        self._model = None
        self._has_weights = False

        if os.path.exists(weights_path):
            self._load_model(weights_path)
        elif not demo_mode:
            logger.warning(
                "YOLO11n weights not found at %s. "
                "Place trained weights at this path for production inference.",
                weights_path,
            )
        else:
            logger.info(
                "YOLO11nDetector: demo mode — no weights loaded. "
                "predict() will return empty detections."
            )

        logger.info(
            "YOLO11nDetector (CASE C Native 1C): classes=%d, conf=%.2f, weights=%s, demo=%s",
            num_classes, conf_threshold,
            "loaded" if self._has_weights else "none",
            demo_mode,
        )

    def _load_model(self, weights_path: str) -> None:
        """
        Load checkpoint and verify/adapt the first convolution to native 1-channel.
        """
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model = model.float().eval()

            # Inspect and adapt first convolution if needed
            first_conv = model.model[0].conv
            if first_conv.in_channels != 1:
                logger.info(
                    "Adapting first conv from Conv2d(%d, %d, ...) to native Conv2d(1, %d, ...)",
                    first_conv.in_channels, first_conv.out_channels, first_conv.out_channels,
                )
                new_conv = nn.Conv2d(
                    1,
                    first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None,
                )
                # W_new = mean(W, dim=channel)
                new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data
                model.model[0].conv = new_conv

            # Verify native 1C structure
            assert model.model[0].conv.in_channels == 1, "First conv must have in_channels=1"
            assert model.model[0].conv.weight.shape[1] == 1, "First conv weight shape must be [C_out, 1, k, k]"

            self._model = model
            self._has_weights = True
            logger.info(
                "YOLO11nDetector: Loaded native 1-channel model from %s. First conv: %s",
                weights_path, model.model[0].conv,
            )
        except Exception as e:
            logger.error("Failed to load YOLO11n weights from %s: %s", weights_path, e)
            if not self.demo_mode:
                raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw forward pass — not supported directly on YOLO11nDetector wrapper.
        Use predict() for inference with NMS and Detection object decoding.
        """
        raise NotImplementedError(
            "YOLO11nDetector does not support raw forward(). "
            "Use predict() for inference."
        )

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        frame_id: str = "unknown",
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run detector inference on native single-channel sonar tensor [1, 1, H, W].

        Args:
            x:              [1, 1, H, W] native single-channel sonar tensor.
            frame_id:       Frame identifier for returned Detection objects.
            conf_threshold: Optional override for confidence threshold.
            iou_threshold:  Optional override for NMS IoU threshold.

        Returns:
            List[Detection] with candidate detections.
        """
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        effective_iou = iou_threshold if iou_threshold is not None else self.nms_iou

        if self._model is None:
            if self.demo_mode:
                return []
            logger.warning("YOLO11nDetector: no weights loaded and demo_mode=False.")
            return []

        # Validate input channel dimensionality
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
        if x.shape[1] != 1:
            raise ValueError(
                f"YOLO11n-1C requires native single-channel input [B, 1, H, W], "
                f"but received tensor with shape {list(x.shape)}."
            )

        # Move input to model device and dtype
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        # Native 1-channel forward pass
        preds = self._model(x)

        # Ultralytics Non-Maximum Suppression
        results = nms.non_max_suppression(
            preds,
            conf_thres=effective_conf,
            iou_thres=effective_iou,
        )

        detections: List[Detection] = []
        if results and len(results) > 0 and len(results[0]) > 0:
            for i, det in enumerate(results[0]):
                x1, y1, x2, y2, conf, cls_id = det.tolist()
                class_id = int(cls_id)
                class_name = self.CLASS_NAMES.get(class_id, "target")

                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                        ),
                        detector_confidence=float(conf),
                        class_id=class_id,
                        class_name=class_name,
                        frame_id=frame_id,
                        detection_id=f"{frame_id}_{i}",
                    )
                )

        logger.debug(
            "Frame %s (native 1C): %d candidate(s) above conf=%.2f",
            frame_id, len(detections), effective_conf,
        )
        return detections

    def load_weights(self, weights_path: str) -> None:
        """Load pretrained weights from a .pt file."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. "
                "Place weights in models/yolo11/weights/."
            )
        self._load_model(weights_path)

    @classmethod
    def from_config(cls, detector_config: Dict, **overrides) -> "YOLO11nDetector":
        """Construct model from a config dict."""
        kwargs = {
            "weights_path": detector_config.get("weights", "models/yolo11/weights/yolo11n_1c.pt"),
            "num_classes": detector_config.get("num_classes", 4),
            "conf_threshold": detector_config.get("confidence_threshold", 0.25),
            "nms_iou": detector_config.get("nms_iou", 0.70),
            "image_size": detector_config.get("image_size", 640),
            "in_channels": detector_config.get("input_channels", 1),
            "demo_mode": detector_config.get("demo_mode", False),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
