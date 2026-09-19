"""
tests/test_pipeline.py — Integration tests for the full pipeline modules.
"""
import pytest
import numpy as np
import torch

from inference.crop_extractor import CropExtractor
from inference.types import BoundingBox, Detection, VerificationResult


class TestCropExtractor:
    def test_extract_returns_tensor(self):
        extractor = CropExtractor(crop_size=128, bbox_padding=0.15)
        frame = np.random.rand(480, 640).astype(np.float32)
        bbox = BoundingBox(100, 100, 200, 200)
        tensor, path = extractor.extract(frame, bbox)
        assert tensor.shape == (1, 1, 128, 128)
        assert tensor.dtype == torch.float32

    def test_extract_with_save(self, tmp_path):
        extractor = CropExtractor(
            crop_size=64, bbox_padding=0.1,
            save_dir=str(tmp_path),
        )
        frame = np.random.rand(200, 300).astype(np.float32)
        bbox = BoundingBox(50, 50, 150, 150)
        tensor, path = extractor.extract(frame, bbox, detection_id="test_001")
        assert tensor.shape == (1, 1, 64, 64)
        assert path is not None
        assert "test_001" in path

    def test_extract_edge_bbox(self):
        extractor = CropExtractor(crop_size=64)
        frame = np.random.rand(100, 100).astype(np.float32)
        bbox = BoundingBox(0, 0, 10, 10)
        tensor, _ = extractor.extract(frame, bbox)
        assert tensor.shape == (1, 1, 64, 64)


class TestEvidenceFusion:
    def test_fuse_returns_detection_result(self):
        from inference.evidence_fusion import EvidenceFusion
        from inference.types import DetectionResult
        from metadata.schema import FrameMetadata

        fusion = EvidenceFusion({
            "artificialness": {
                "detector_weight": 0.35, "verifier_weight": 0.45,
                "contrast_weight": 0.10, "geometry_weight": 0.10,
            },
            "priority": {
                "artificialness_weight": 0.70,
                "mission_confidence_weight": 0.20,
                "metadata_quality_weight": 0.10,
            },
        })

        det = Detection(
            bbox=BoundingBox(100, 100, 200, 200),
            detector_confidence=0.85,
            class_id=0,
            class_name="target",
            frame_id="f1",
        )
        ver = VerificationResult(
            natural_probability=0.3,
            artificial_probability=0.7,
            predicted_class="artificial",
        )
        frame = np.random.rand(480, 640).astype(np.float32)
        meta = FrameMetadata(frame_id="f1", latitude=42.0, longitude=-71.0)

        result = fusion.fuse(det, ver, frame, meta, detection_id="test_001")

        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.artificialness_score <= 1.0
        assert 0.0 <= result.priority_score <= 1.0
        assert result.verifier_class == "artificial"
        assert result.latitude == 42.0


class TestReportGeneration:
    def test_json_export(self, tmp_path):
        from reporting.json_export import JSONExporter
        from inference.types import (
            MissionResult, DetectionResult, BoundingBox, DetectionStatus,
        )

        exporter = JSONExporter(output_dir=str(tmp_path))
        results = [
            MissionResult(
                frame_id="f1",
                mission_id="test",
                detections=[
                    DetectionResult(
                        detection_id="d1",
                        bbox=BoundingBox(10, 20, 100, 200),
                        detector_confidence=0.9,
                        detector_class="target",
                        crop_path="",
                        verifier_class="artificial",
                        artificial_probability=0.8,
                        natural_probability=0.2,
                        artificialness_score=0.85,
                        priority_score=0.75,
                        latitude=None,
                        longitude=None,
                        timestamp=None,
                        heading=None,
                        ping_id=None,
                        status=DetectionStatus.VERIFIED,
                    ),
                ],
            ),
        ]

        path = exporter.export_mission(results, mission_id="test")
        assert path.endswith(".json")

    def test_csv_export(self, tmp_path):
        from reporting.csv_export import CSVExporter
        from inference.types import (
            MissionResult, DetectionResult, BoundingBox, DetectionStatus,
        )

        exporter = CSVExporter(output_dir=str(tmp_path))
        results = [
            MissionResult(
                frame_id="f1",
                mission_id="test",
                detections=[
                    DetectionResult(
                        detection_id="d1",
                        bbox=BoundingBox(10, 20, 100, 200),
                        detector_confidence=0.9,
                        detector_class="target",
                        crop_path="",
                        verifier_class="artificial",
                        artificial_probability=0.8,
                        natural_probability=0.2,
                        artificialness_score=0.85,
                        priority_score=0.75,
                        latitude=None,
                        longitude=None,
                        timestamp=None,
                        heading=None,
                        ping_id=None,
                        status=DetectionStatus.VERIFIED,
                    ),
                ],
            ),
        ]

        path = exporter.export_mission(results, mission_id="test")
        assert path.endswith(".csv")
