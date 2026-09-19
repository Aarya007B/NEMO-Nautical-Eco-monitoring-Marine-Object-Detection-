"""
metadata/schema.py — Data schemas for mission metadata.

These schemas define the structure of GPS, ping, and frame metadata
used to geo-reference sonar detections.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NavigationRecord:
    """Single GPS/navigation record from the AUV or vessel."""
    timestamp: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0
    depth: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0


@dataclass
class PingMetadata:
    """Sonar ping-level metadata."""
    ping_number: int = 0
    timestamp: float = 0.0
    range_m: float = 0.0
    frequency_khz: float = 0.0
    gain_db: float = 0.0
    slant_range_correction: bool = False


@dataclass
class FrameMetadata:
    """Combined metadata for a single sonar frame."""
    frame_id: str = ""
    mission_id: str = ""
    timestamp: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading: Optional[float] = None
    depth: Optional[float] = None
    ping: Optional[PingMetadata] = None
    navigation: Optional[NavigationRecord] = None
    image_path: str = ""
    image_width: int = 0
    image_height: int = 0

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def has_timestamp(self) -> bool:
        return self.timestamp is not None
