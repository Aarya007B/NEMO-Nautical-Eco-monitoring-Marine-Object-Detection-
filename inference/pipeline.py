"""
inference/pipeline.py — Full two-stage mission pipeline.

Orchestrates:
    1. Preprocessing
    2. Stage-1 RCDI-YOLO detection
    3. Crop extraction
    4. Stage-2 MobileNetV3 verification
    5. Evidence fusion
    6. Metadata alignment

Usage:
    pipeline = MissionPipeline.from_config(config)
    for frame in mission_source:
        result = pipeline.process(frame)
"""
import logging
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np

from inference.crop_extractor import CropExtractor
from inference.detector import CandidateDetector
from inference.evidence_fusion import EvidenceFusion
from inference.types import DetectionResult, MissionFrame, MissionResult
from inference.verifier import CandidateVerifier
from metadata.alignment import MetadataAligner
from metadata.gps import GPSParser
from metadata.schema import FrameMetadata
from preprocessing.normalize import to_float32

logger = logging.getLogger(__name__)


class MissionPipeline:
    """
    End-to-end two-stage sonar detection pipeline.

    Processes MissionFrame objects and returns MissionResult objects
    containing all detections with scores and metadata.
    """

    def __init__(
        self,
        detector: CandidateDetector,
        verifier: CandidateVerifier,
        crop_extractor: CropExtractor,
        evidence_fusion: EvidenceFusion,
        metadata_aligner: Optional[MetadataAligner] = None,
    ):
        self.detector = detector
        self.verifier = verifier
        self.crop_extractor = crop_extractor
        self.evidence_fusion = evidence_fusion
        self.metadata_aligner = metadata_aligner or MetadataAligner()

    @classmethod
    def from_config(
        cls,
        config: Dict,
        navigation_file: Optional[str] = None,
    ) -> "MissionPipeline":
        """
        Build pipeline from config.yaml.

        Args:
            config: Parsed config dict.
            navigation_file: Optional path to navigation CSV.

        Returns:
            Configured MissionPipeline instance.
        """
        device = config.get("runtime", {}).get("device", "auto")

        detector = CandidateDetector(
            detector_config=config.get("detector", {}),
            preprocessing_config=config.get("preprocessing", {}),
            device=device,
        )

        verifier = CandidateVerifier(
            verifier_config=config.get("verifier", {}),
            device=device,
        )

        output_config = config.get("output", {})
        crop_extractor = CropExtractor(
            crop_size=config.get("verifier", {}).get("crop_size", 128),
            bbox_padding=config.get("verifier", {}).get("bbox_padding", 0.15),
            save_dir=output_config.get("directory", "outputs/") + "crops/"
                if output_config.get("save_crops", True) else None,
        )

        evidence_fusion = EvidenceFusion(
            scoring_config=config.get("scoring", {}),
        )

        # Load navigation data if available
        aligner = MetadataAligner()
        if navigation_file:
            parser = GPSParser()
            nav_records = parser.parse(navigation_file)
            if nav_records:
                aligner = MetadataAligner(nav_records=nav_records)
                logger.info("Loaded %d navigation records", len(nav_records))

        return cls(
            detector=detector,
            verifier=verifier,
            crop_extractor=crop_extractor,
            evidence_fusion=evidence_fusion,
            metadata_aligner=aligner,
        )

    def process(self, frame: MissionFrame) -> MissionResult:
        """
        Process a single mission frame through the full pipeline.

        Args:
            frame: MissionFrame with image_path and metadata.

        Returns:
            MissionResult with all detections and scores.
        """
        # Load image
        image = cv2.imread(frame.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error("Failed to load image: %s", frame.image_path)
            return MissionResult(
                frame_id=frame.frame_id,
                mission_id=frame.mission_id,
                detections=[],
            )

        image_f32 = to_float32(image)

        # Stage 1: Detection
        detections, _, scale, padding = self.detector.detect(
            image, frame_id=frame.frame_id,
        )

        # Align metadata
        frame_meta = self.metadata_aligner.align(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )
        frame_meta.image_path = frame.image_path
        frame_meta.mission_id = frame.mission_id
        frame_meta.image_height, frame_meta.image_width = image.shape[:2]

        # Process each detection through Stage 2
        results: List[DetectionResult] = []
        for i, det in enumerate(detections):
            det_id = f"{frame.frame_id}_{i:03d}_{uuid.uuid4().hex[:6]}"

            # Extract crop
            crop_tensor, crop_path = self.crop_extractor.extract(
                image_f32, det.bbox, detection_id=det_id,
            )

            # Stage 2: Verification
            verification = self.verifier.verify(crop_tensor)

            # Fuse evidence
            result = self.evidence_fusion.fuse(
                detection=det,
                verification=verification,
                frame=image_f32,
                frame_metadata=frame_meta,
                crop_path=crop_path or "",
                detection_id=det_id,
            )
            results.append(result)

        logger.debug(
            "Frame %s: %d candidates, %d verified",
            frame.frame_id, len(detections),
            sum(1 for r in results if r.status.value == "verified"),
        )

        return MissionResult(
            frame_id=frame.frame_id,
            mission_id=frame.mission_id,
            detections=results,
            frame_metadata=frame_meta,
        )

    def process_mission(
        self,
        source,
        progress_callback=None,
    ) -> List[MissionResult]:
        """
        Process an entire mission.

        Args:
            source: MissionSource to iterate.
            progress_callback: Optional callback(frame_idx, total_frames).

        Returns:
            List of MissionResult for every frame.
        """
        all_results = []
        total = len(source)

        for idx, frame in enumerate(source):
            result = self.process(frame)
            all_results.append(result)

            if progress_callback:
                progress_callback(idx + 1, total)

        logger.info(
            "Mission %s complete: %d frames, %d total detections",
            source.mission_id, total,
            sum(len(r.detections) for r in all_results),
        )
        return all_results
