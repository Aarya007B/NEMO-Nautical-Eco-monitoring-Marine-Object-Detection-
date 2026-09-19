"""
tests/test_metadata.py — Tests for metadata alignment module.
"""
import pytest

from metadata.schema import NavigationRecord, PingMetadata, FrameMetadata
from metadata.alignment import MetadataAligner


class TestNavigationRecord:
    def test_defaults(self):
        rec = NavigationRecord()
        assert rec.timestamp == 0.0
        assert rec.latitude == 0.0

    def test_custom_values(self):
        rec = NavigationRecord(latitude=42.5, longitude=-71.3, heading=180.0)
        assert rec.latitude == 42.5
        assert rec.heading == 180.0


class TestFrameMetadata:
    def test_has_gps(self):
        meta = FrameMetadata(latitude=42.0, longitude=-71.0)
        assert meta.has_gps is True

    def test_no_gps(self):
        meta = FrameMetadata()
        assert meta.has_gps is False

    def test_has_timestamp(self):
        meta = FrameMetadata(timestamp=1234567890.0)
        assert meta.has_timestamp is True


class TestMetadataAligner:
    def test_no_nav_records(self):
        aligner = MetadataAligner()
        meta = aligner.align("frame_001", timestamp=1000.0)
        assert meta.frame_id == "frame_001"
        assert meta.latitude is None

    def test_exact_match(self):
        records = [
            NavigationRecord(timestamp=1000.0, latitude=42.0, longitude=-71.0),
            NavigationRecord(timestamp=2000.0, latitude=43.0, longitude=-72.0),
        ]
        aligner = MetadataAligner(nav_records=records)
        meta = aligner.align("frame_001", timestamp=1000.0)
        assert meta.latitude == 42.0
        assert meta.longitude == -71.0

    def test_nearest_match(self):
        records = [
            NavigationRecord(timestamp=1000.0, latitude=42.0, longitude=-71.0),
            NavigationRecord(timestamp=1005.0, latitude=43.0, longitude=-72.0),
        ]
        # max_time_gap=10 so both are within range
        aligner = MetadataAligner(nav_records=records, max_time_gap=10.0)
        meta = aligner.align("frame_001", timestamp=1002.0)
        # Should match nearest (1000.0 is 2s away, 1005.0 is 3s away -> picks 42.0)
        assert meta.latitude == 42.0

    def test_max_gap_exceeded(self):
        records = [
            NavigationRecord(timestamp=1000.0, latitude=42.0, longitude=-71.0),
        ]
        aligner = MetadataAligner(nav_records=records, max_time_gap=5.0)
        meta = aligner.align("frame_001", timestamp=2000.0)
        assert meta.latitude is None

    def test_no_timestamp(self):
        records = [NavigationRecord(timestamp=1000.0, latitude=42.0)]
        aligner = MetadataAligner(nav_records=records)
        meta = aligner.align("frame_001", timestamp=None)
        assert meta.latitude is None

    def test_batch_align(self):
        records = [
            NavigationRecord(timestamp=100.0, latitude=10.0, longitude=20.0),
            NavigationRecord(timestamp=200.0, latitude=11.0, longitude=21.0),
        ]
        aligner = MetadataAligner(nav_records=records)
        results = aligner.align_batch(
            ["f1", "f2"], [100.0, 200.0],
        )
        assert len(results) == 2
        assert results[0].latitude == 10.0
        assert results[1].latitude == 11.0
