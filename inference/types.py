"""
inference/types.py — Shared typed data structures for NEMO.

All major subsystems communicate through these typed dataclasses.
Avoid passing raw dicts between modules; use these types instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Detection status
# ---------------------------------------------------------------------------

class DetectionStatus(str, Enum):
    """Human-verification lifecycle state of a detection."""
    CANDIDATE = "candidate"   # detector output, not yet verified
    VERIFIED  = "verified"    # verifier: artificial + optionally human-confirmed
    REJECTED  = "rejected"    # verifier: natural / dismissed
    REVIEWED  = "reviewed"    # human operator has manually reviewed


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float   # left
    y1: float   # top
    x2: float   # right
    y2: float   # bottom

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)

    def expand(self, padding: float, img_w: float, img_h: float) -> "BoundingBox":
        """Expand bbox by a fractional padding and clip to image bounds."""
        pw = self.width * padding
        ph = self.height * padding
        return BoundingBox(
            x1=max(0.0, self.x1 - pw),
            y1=max(0.0, self.y1 - ph),
            x2=min(img_w, self.x2 + pw),
            y2=min(img_h, self.y2 + ph),
        )

    def to_xywh(self):
        return (self.x1, self.y1, self.width, self.height)

    def to_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)


# ---------------------------------------------------------------------------
# Detector output
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Raw output from the Stage-1 candidate detector (YOLO11n-1C in validated MVP).

    Stores detector-level evidence independently from the verifier output
    so scores remain traceable (PRD §13).
    """
    bbox: BoundingBox
    detector_confidence: float   # sigmoid score from detector head
    class_id: int
    class_name: str
    frame_id: str
    detection_id: str = ""


# ---------------------------------------------------------------------------
# Verifier output
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """
    Output from the MobileNetV3-Small Stage-2 verifier.

    natural_probability + artificial_probability should sum to ~1.0.
    Stored independently from detector_confidence (PRD §13).
    """
    natural_probability: float
    artificial_probability: float
    predicted_class: str          # "natural" or "artificial"
    raw_logits: Optional[List[float]] = None

    def is_artificial(self, threshold: float = 0.5) -> bool:
        return self.artificial_probability >= threshold


# ---------------------------------------------------------------------------
# Acoustic feature evidence
# ---------------------------------------------------------------------------

@dataclass
class AcousticFeatures:
    """
    Sonar-domain evidence features extracted from the raw detection crop.

    Used by EvidenceFusion in addition to model probabilities.
    All fields default to neutral values when computation fails gracefully.
    """
    local_contrast: float = 0.5          # contrast within bbox region [0,1]
    intensity_difference: float = 0.5   # target vs background intensity [0,1]
    bounding_box_area: float = 0.0      # normalized area [0,1]
    aspect_ratio: float = 1.0           # width / height
    edge_density: float = 0.5           # Canny edge density in crop [0,1]
    image_quality: float = 1.0          # overall frame quality estimate [0,1]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """
    Combined evidence fusion output.

    artificialness_score: how strongly evidence suggests man-made object.
    priority_score: how important this detection is for operator review.
    These are DIFFERENT concepts (PRD §25) and must not be conflated.
    """
    artificialness_score: float          # [0, 1]
    priority_score: float                # [0, 1]
    component_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Full pipeline detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Complete detection record after running the full two-stage pipeline.

    Contains independently stored detector and verifier evidence (PRD §13).
    """
    detection_id: str
    bbox: BoundingBox

    # Stage 1 — YOLO11n-1C detector (validated MVP)
    detector_confidence: float
    detector_class: str

    # Stage 2 — MobileNetV3 verifier
    crop_path: Optional[str]
    verifier_class: Optional[str]
    artificial_probability: Optional[float]
    natural_probability: Optional[float]

    # Evidence fusion scores
    artificialness_score: float
    priority_score: float

    # Geolocation
    latitude: Optional[float]
    longitude: Optional[float]
    timestamp: Optional[str]
    heading: Optional[float]
    ping_id: Optional[str]

    # Lifecycle
    status: DetectionStatus = DetectionStatus.CANDIDATE
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mission frame
# ---------------------------------------------------------------------------

@dataclass
class MissionFrame:
    """
    Unified sonar frame interface for both recorded and live modes (PRD §21).

    Both RecordedMissionSource and LiveMissionSource produce MissionFrames
    so the inference pipeline never needs to know the input mode.
    """
    mission_id: str = ""
    frame_id: str = ""
    sonar_path: str = ""
    image_path: str = ""
    frame_index: int = 0
    timestamp: Optional[Any] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading: Optional[float] = None
    ping_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.image_path and self.sonar_path:
            self.image_path = self.sonar_path
        elif not self.sonar_path and self.image_path:
            self.sonar_path = self.image_path


# ---------------------------------------------------------------------------
# Mission result
# ---------------------------------------------------------------------------

@dataclass
class MissionResult:
    """Output of MissionPipeline.process() for a single frame."""
    mission_id: str = ""
    frame_id: str = ""
    detections: List[DetectionResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    frame_metadata: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset record (data adapter)
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecord:
    """
    Canonical dataset record produced by every dataset adapter (PRD §16).

    Preserves source metadata alongside canonical class mapping.
    """
    image_path: str
    annotation_path: Optional[str]
    source_dataset: str
    source_class: str
    canonical_class: str          # always "target" for Stage-1 detector
    mission_id: Optional[str] = None
    frame_id: Optional[str] = None
    ping_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[str] = None
    heading: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
