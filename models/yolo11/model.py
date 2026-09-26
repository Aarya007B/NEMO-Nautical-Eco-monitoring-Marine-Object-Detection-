"""
models/yolo11/model.py

YOLO11nDetector — NEMO validated MVP detector.

This is the experimentally validated Stage-1 detector selected based on
T1/T2/T3/T4 ablation experiments. It uses native 1-channel sonar input
at 640×640 resolution.

Architecture: YOLO11n (Ultralytics) adapted for 1-channel input.
Input:        [B, 1, 640, 640] native sonar acoustic-intensity.
Output:       List[Detection] with bounding boxes, confidence, and class.

USAGE:
    model = YOLO11nDetector(weights_path="models/yolo11/weights/yolo11n_1c.pt")
    detections = model.predict(tensor)  # [1, 1, 640, 640]
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from inference.types import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YOLO11nDetector(nn.Module):
    """
    YOLO11n-1C detector for native 1-channel sonar images.

    Wraps a YOLO11n model (Ultralytics checkpoint) to produce Detection
    objects compatible with the NEMO inference pipeline.

    In production mode, requires trained weights — raises an error if
    weights are not found and demo_mode is not enabled.

    In demo mode (demo_mode=True), returns empty detections when no
    weights are available, allowing pipeline testing without trained models.

    Args:
        weights_path:     Path to .pt weights file.
        num_classes:      Number of detection classes (default 1).
        conf_threshold:   Confidence threshold for filtering.
        nms_iou:          NMS IoU threshold.
        image_size:       Expected input size.
        in_channels:      Input channels (1 for sonar).
        demo_mode:        If True, allow running without weights (empty output).
    """

    def __init__(
        self,
        weights_path: str = "models/yolo11/weights/yolo11n_1c.pt",
        num_classes: int = 1,
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
        self._ultralytics_model = None
        self._has_weights = False

        # Attempt to load via ultralytics
        if os.path.exists(weights_path):
            self._load_ultralytics(weights_path)
        elif not demo_mode:
            logger.warning(
                "YOLO11n weights not found at %s. "
                "Place trained weights at this path for production inference. "
                "To run without weights, set demo_mode=True in config.",
                weights_path,
            )
        else:
            logger.info(
                "YOLO11nDetector: demo mode — no weights loaded. "
                "predict() will return empty detections."
            )

        logger.info(
            "YOLO11nDetector: classes=%d, conf=%.2f, weights=%s, demo=%s",
            num_classes, conf_threshold,
            "loaded" if self._has_weights else "none",
            demo_mode,
        )

    def _load_ultralytics(self, weights_path: str) -> None:
        """Load model via ultralytics package."""
        try:
            from ultralytics import YOLO
            self._ultralytics_model = YOLO(weights_path)
            self._has_weights = True
            logger.info(
                "YOLO11nDetector: Loaded weights from %s (ultralytics)",
                weights_path,
            )
        except ImportError:
            logger.warning(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.warning(
                "Failed to load YOLO11n weights from %s: %s",
                weights_path, e,
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Raw forward pass — not supported for ultralytics-backed models.
        Use predict() instead.
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
        Run detector inference and return decoded detections.

        Args:
            x:              [1, 1, H, W] sonar tensor (batch size 1).
            frame_id:       Frame identifier for returned Detection objects.
            conf_threshold: Optional override for confidence threshold.
            iou_threshold:  Optional override for NMS IoU threshold.

        Returns:
            List[Detection] — may be empty if no detections pass threshold
            or if running in demo mode without weights.
        """
        effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        effective_iou = iou_threshold if iou_threshold is not None else self.nms_iou

        if self._ultralytics_model is not None:
            return self._predict_ultralytics(x, frame_id, effective_conf, effective_iou)
        elif self.demo_mode:
            # Demo mode: return empty detections
            return []
        else:
            logger.warning(
                "YOLO11nDetector: no weights loaded and demo_mode=False. "
                "Returning empty detections. Load weights for production use."
            )
            return []

    def _predict_ultralytics(
        self,
        x: torch.Tensor,
        frame_id: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Detection]:
        """Predict using the loaded ultralytics model."""
        # Convert 1-channel tensor to numpy for ultralytics
        img_np = x[0, 0].cpu().numpy()  # [H, W]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        results = self._ultralytics_model.predict(
            img_np,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf[0].cpu())
                    cls_id = int(box.cls[0].cpu())

                    detections.append(
                        Detection(
                            bbox=BoundingBox(
                                x1=xyxy[0], y1=xyxy[1],
                                x2=xyxy[2], y2=xyxy[3],
                            ),
                            detector_confidence=conf,
                            class_id=cls_id,
                            class_name="target",
                            frame_id=frame_id,
                            detection_id=f"{frame_id}_{i}",
                        )
                    )

        logger.debug(
            "Frame %s: %d candidate(s) above conf=%.2f",
            frame_id, len(detections), conf_threshold,
        )
        return detections

    def load_weights(self, weights_path: str) -> None:
        """
        Load pretrained weights from a .pt file.

        Args:
            weights_path: Path to weight file.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. "
                "Place weights in models/yolo11/weights/."
            )
        self._load_ultralytics(weights_path)

    @classmethod
    def from_config(cls, detector_config: Dict, **overrides) -> "YOLO11nDetector":
        """
        Construct model from a config dict.

        Args:
            detector_config: Detector config dict from config.yaml.
            **overrides: Any constructor argument overrides.

        Returns:
            Initialized YOLO11nDetector.
        """
        kwargs = {
            "weights_path": detector_config.get("weights", "models/yolo11/weights/yolo11n_1c.pt"),
            "num_classes": detector_config.get("num_classes", 1),
            "conf_threshold": detector_config.get("confidence_threshold", 0.25),
            "nms_iou": detector_config.get("nms_iou", 0.70),
            "image_size": detector_config.get("image_size", 640),
            "in_channels": detector_config.get("input_channels", 1),
            "demo_mode": detector_config.get("demo_mode", False),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
