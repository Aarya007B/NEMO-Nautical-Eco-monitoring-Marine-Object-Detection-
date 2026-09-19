"""
metadata/alignment.py — Frame-to-navigation metadata alignment.

Aligns sonar frames with GPS/navigation records using timestamp
interpolation.
"""
import bisect
import logging
from typing import List, Optional

from metadata.schema import FrameMetadata, NavigationRecord, PingMetadata

logger = logging.getLogger(__name__)


class MetadataAligner:
    """
    Aligns sonar frames with navigation records.

    Uses timestamp-based nearest-neighbor matching with optional
    linear interpolation for GPS coordinates.
    """

    def __init__(
        self,
        nav_records: Optional[List[NavigationRecord]] = None,
        max_time_gap: float = 10.0,
    ):
        """
        Args:
            nav_records: Sorted list of NavigationRecord by timestamp.
            max_time_gap: Maximum allowable time gap (seconds) for matching.
        """
        self.nav_records = sorted(nav_records or [], key=lambda r: r.timestamp)
        self.nav_timestamps = [r.timestamp for r in self.nav_records]
        self.max_time_gap = max_time_gap

    def align(
        self,
        frame_id: str,
        timestamp: Optional[float] = None,
        ping: Optional[PingMetadata] = None,
    ) -> FrameMetadata:
        """
        Align a frame with the closest navigation record.

        Args:
            frame_id: Unique frame identifier.
            timestamp: Frame capture timestamp (epoch seconds).
            ping: Optional ping metadata.

        Returns:
            FrameMetadata with GPS populated if a match was found.
        """
        meta = FrameMetadata(frame_id=frame_id, timestamp=timestamp, ping=ping)

        if timestamp is None or not self.nav_records:
            return meta

        # Find nearest navigation record
        idx = bisect.bisect_left(self.nav_timestamps, timestamp)

        # Check candidates on both sides
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self.nav_records):
            candidates.append(idx)

        best_idx = min(candidates, key=lambda i: abs(self.nav_timestamps[i] - timestamp))
        time_gap = abs(self.nav_timestamps[best_idx] - timestamp)

        if time_gap > self.max_time_gap:
            logger.debug(
                "Frame %s: nearest nav record %.1fs away (max %.1fs), skipping",
                frame_id, time_gap, self.max_time_gap,
            )
            return meta

        nav = self.nav_records[best_idx]
        meta.latitude = nav.latitude
        meta.longitude = nav.longitude
        meta.heading = nav.heading
        meta.depth = nav.depth
        meta.navigation = nav

        return meta

    def align_batch(
        self,
        frame_ids: List[str],
        timestamps: List[Optional[float]],
    ) -> List[FrameMetadata]:
        """Align multiple frames at once."""
        return [
            self.align(fid, ts)
            for fid, ts in zip(frame_ids, timestamps)
        ]
