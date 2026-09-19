"""
inference — SonarGuard two-stage detection pipeline.

Exports:
    Types:      BoundingBox, Detection, VerificationResult, etc.
    Pipeline:   CandidateDetector, CandidateVerifier, CropExtractor,
                EvidenceFusion, MissionPipeline
"""
from inference.types import (
    DetectionStatus,
    BoundingBox,
    Detection,
    VerificationResult,
    AcousticFeatures,
    ScoreResult,
    DetectionResult,
    MissionFrame,
    MissionResult,
    DatasetRecord,
)

__all__ = [
    # Types
    "DetectionStatus",
    "BoundingBox",
    "Detection",
    "VerificationResult",
    "AcousticFeatures",
    "ScoreResult",
    "DetectionResult",
    "MissionFrame",
    "MissionResult",
    "DatasetRecord",
]
