"""
inference/detector.py — CandidateDetector wrapper.

Wraps the RCDI-YOLO model with preprocessing for use in the
inference pipeline.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.types import Detection, BoundingBox
from models.rcdi_yolo.model import RCDIYOLOModel
from preprocessing.pipeline import SonarPreprocessor

logger = logging.getLogger(__name__)


class CandidateDetector:
    """
    Stage-1 candidate detector.

    Wraps SonarPreprocessor + RCDI-YOLO model to detect candidates
    from raw sonar images.

    Args:
        detector_config: Detector config dict from config.yaml.
        preprocessing_config: Preprocessing config dict from config.yaml.
        device: Target device (auto/cuda/mps/cpu).
    """

    def __init__(
        self,
        detector_config: Dict,
        preprocessing_config: Dict,
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)
        self.preprocessor = SonarPreprocessor(preprocessing_config)

        # Build or load model
        model_config = detector_config.get("config", "configs/rcdi_yolo_1c.yaml")
        self.model = RCDIYOLOModel.from_config(model_config).to(self.device)
        self.model.eval()

        # Attempt to load weights
        weights_path = detector_config.get("weights", "")
        if weights_path:
            try:
                self.model.load_weights(weights_path)
                logger.info("Loaded detector weights: %s", weights_path)
            except FileNotFoundError:
                logger.warning(
                    "Detector weights not found at %s. "
                    "Running with random weights (smoke test mode).",
                    weights_path,
                )

        self.conf_threshold = detector_config.get("confidence_threshold", 0.25)
        self.nms_iou = detector_config.get("nms_iou", 0.70)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "CandidateDetector: device=%s, params=%.2fM, conf=%.2f",
            self.device, total_params / 1e6, self.conf_threshold,
        )

    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        frame_id: str = "",
    ) -> Tuple[List[Detection], np.ndarray, float, Tuple[int, int]]:
        """
        Run detection on a sonar image.

        Args:
            image: Raw sonar image (uint8 or float32, any shape).
            frame_id: Frame identifier for traceability.

        Returns:
            Tuple of (detections, preprocessed_frame, scale, padding).
        """
        tensor, scale, padding = self.preprocessor(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        detections = self.model.predict(
            tensor, frame_id=frame_id,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.nms_iou,
        )

        return detections, image, scale, padding

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
