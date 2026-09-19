"""metadata — GPS, ping, and navigation metadata subsystem."""
from metadata.schema import NavigationRecord, PingMetadata, FrameMetadata
from metadata.gps import GPSParser
from metadata.ping import PingParser
from metadata.alignment import MetadataAligner
__all__ = [
    "NavigationRecord", "PingMetadata", "FrameMetadata",
    "GPSParser", "PingParser", "MetadataAligner",
]
