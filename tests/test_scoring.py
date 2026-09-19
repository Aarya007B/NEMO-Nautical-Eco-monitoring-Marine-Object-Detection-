"""
tests/test_scoring.py — Tests for scoring module.
"""
import pytest
import numpy as np

from inference.types import AcousticFeatures, BoundingBox
from scoring.features import AcousticFeatureExtractor
from scoring.artificialness import ArtificialnessScorer
from scoring.priority import PriorityScorer


class TestAcousticFeatureExtractor:
    def test_extract_returns_features(self):
        extractor = AcousticFeatureExtractor()
        frame = np.random.rand(480, 640).astype(np.float32)
        bbox = BoundingBox(100, 100, 200, 200)
        features = extractor.extract(frame, bbox)
        assert isinstance(features, AcousticFeatures)

    def test_features_normalized(self):
        extractor = AcousticFeatureExtractor()
        frame = np.random.rand(480, 640).astype(np.float32)
        bbox = BoundingBox(50, 50, 150, 150)
        features = extractor.extract(frame, bbox)
        assert 0.0 <= features.local_contrast <= 1.0
        assert 0.0 <= features.edge_density <= 1.0
        assert 0.0 <= features.bounding_box_area <= 1.0

    def test_invalid_bbox(self):
        extractor = AcousticFeatureExtractor()
        frame = np.random.rand(100, 100).astype(np.float32)
        bbox = BoundingBox(0, 0, 0, 0)  # zero-area
        features = extractor.extract(frame, bbox)
        assert isinstance(features, AcousticFeatures)


class TestArtificialnessScorer:
    def test_score_in_range(self):
        scorer = ArtificialnessScorer({
            "detector_weight": 0.35,
            "verifier_weight": 0.45,
            "contrast_weight": 0.10,
            "geometry_weight": 0.10,
        })
        features = AcousticFeatures(local_contrast=0.5, aspect_ratio=1.2)
        score = scorer.score(0.8, 0.9, features)
        assert 0.0 <= score <= 1.0

    def test_high_scores_produce_high_artificialness(self):
        scorer = ArtificialnessScorer({
            "detector_weight": 0.35,
            "verifier_weight": 0.45,
            "contrast_weight": 0.10,
            "geometry_weight": 0.10,
        })
        features = AcousticFeatures(local_contrast=0.9, aspect_ratio=1.0)
        score = scorer.score(0.95, 0.95, features)
        assert score > 0.5

    def test_low_scores_produce_low_artificialness(self):
        scorer = ArtificialnessScorer({
            "detector_weight": 0.35,
            "verifier_weight": 0.45,
            "contrast_weight": 0.10,
            "geometry_weight": 0.10,
        })
        features = AcousticFeatures(local_contrast=0.1, aspect_ratio=5.0)
        score = scorer.score(0.1, 0.05, features)
        assert score < 0.5


class TestPriorityScorer:
    def test_score_in_range(self):
        scorer = PriorityScorer({
            "artificialness_weight": 0.70,
            "mission_confidence_weight": 0.20,
            "metadata_quality_weight": 0.10,
        })
        score = scorer.score(0.8, 0.7, has_gps=True, has_timestamp=True)
        assert 0.0 <= score <= 1.0

    def test_gps_increases_priority(self):
        scorer = PriorityScorer({
            "artificialness_weight": 0.70,
            "mission_confidence_weight": 0.20,
            "metadata_quality_weight": 0.10,
        })
        no_gps = scorer.score(0.8, 0.7, has_gps=False, has_timestamp=False)
        with_gps = scorer.score(0.8, 0.7, has_gps=True, has_timestamp=True)
        assert with_gps >= no_gps
