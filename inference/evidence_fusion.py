"""
inference/evidence_fusion.py — Evidence fusion layer.

Combines Stage-1 detector, Stage-2 verifier, acoustic features, and
metadata into final artificialness_score and priority_score.
"""
import logging
from typing import Dict

from inference.types import (
    AcousticFeatures, BoundingBox, Detection, DetectionResult, DetectionStatus,
    VerificationResult,
)
from metadata.schema import FrameMetadata
from scoring.artificialness import ArtificialnessScorer
from scoring.features import AcousticFeatureExtractor
from scoring.priority import PriorityScorer

import numpy as np

logger = logging.getLogger(__name__)


class EvidenceFusion:
    """
    Fuses evidence from detector, verifier, and acoustic features.

    Produces:
        artificialness_score — how strongly does evidence point to man-made?
        priority_score       — how urgently should an operator review this?

    These are SEPARATE concepts (PRD §25).
    """

    def __init__(self, scoring_config: Dict):
        self.feature_extractor = AcousticFeatureExtractor()
        self.artificialness_scorer = ArtificialnessScorer(
            scoring_config.get("artificialness", {})
        )
        self.priority_scorer = PriorityScorer(
            scoring_config.get("priority", {})
        )
        logger.info("EvidenceFusion initialized")

    def fuse(
        self,
        detection: Detection,
        verification: VerificationResult,
        frame: np.ndarray,
        frame_metadata: FrameMetadata,
        crop_path: str = "",
        detection_id: str = "",
    ) -> DetectionResult:
        """
        Fuse all evidence into a DetectionResult.

        Args:
            detection: Stage-1 detection.
            verification: Stage-2 verification result.
            frame: Preprocessed grayscale frame [H, W] as float32.
            frame_metadata: GPS/navigation metadata.
            crop_path: Path to saved crop image.
            detection_id: Unique detection identifier.

        Returns:
            DetectionResult with all scores populated.
        """
        features = self.feature_extractor.extract(frame, detection.bbox)

        artificialness = self.artificialness_scorer.score(
            detector_confidence=detection.detector_confidence,
            artificial_probability=verification.artificial_probability,
            features=features,
        )

        priority = self.priority_scorer.score(
            artificialness_score=artificialness,
            detector_confidence=detection.detector_confidence,
            has_gps=frame_metadata.has_gps,
            has_timestamp=frame_metadata.has_timestamp,
        )

        # Determine status
        if verification.predicted_class == "artificial":
            status = DetectionStatus.VERIFIED
        else:
            status = DetectionStatus.REJECTED

        return DetectionResult(
            detection_id=detection_id,
            bbox=detection.bbox,
            detector_confidence=detection.detector_confidence,
            detector_class=detection.class_name,
            crop_path=crop_path,
            verifier_class=verification.predicted_class,
            artificial_probability=verification.artificial_probability,
            natural_probability=verification.natural_probability,
            artificialness_score=artificialness,
            priority_score=priority,
            latitude=frame_metadata.latitude,
            longitude=frame_metadata.longitude,
            timestamp=frame_metadata.timestamp,
            heading=frame_metadata.heading,
            ping_id=frame_metadata.ping.ping_number if frame_metadata.ping else None,
            status=status,
        )
