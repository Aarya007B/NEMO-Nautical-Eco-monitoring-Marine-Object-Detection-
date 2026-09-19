"""
mission/live.py — Live sonar stream source (future mode).

This is a stub for the future live/streaming sonar data mode.
The MVP uses RecordedMissionSource only.
"""
import logging
from typing import Iterator

from inference.types import MissionFrame
from mission.source import MissionSource

logger = logging.getLogger(__name__)


class LiveMissionSource(MissionSource):
    """
    Live sonar stream source — NOT YET IMPLEMENTED.

    Future implementation will:
        - Connect to a sonar hardware driver or network socket.
        - Yield MissionFrame objects as pings arrive in real time.
        - Support backpressure and frame dropping if processing is slow.
    """

    def __init__(self, **kwargs):
        logger.warning(
            "LiveMissionSource is a stub. "
            "Live mode is planned for a future release."
        )
        self._mission_id = kwargs.get("mission_id", "live_session")

    @property
    def mission_id(self) -> str:
        return self._mission_id

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[MissionFrame]:
        raise NotImplementedError(
            "LiveMissionSource is not yet implemented. "
            "Use RecordedMissionSource for offline processing."
        )
