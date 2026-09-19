"""
mission/source.py — Abstract base class for mission frame sources.

All mission sources iterate over sonar frames in capture order.
"""
from abc import ABC, abstractmethod
from typing import Iterator

from inference.types import MissionFrame


class MissionSource(ABC):
    """
    Abstract base for mission frame sources.

    Implementations:
        RecordedMissionSource — iterate over saved sonar images on disk.
        LiveMissionSource     — stream from a live sonar feed (future).
    """

    @abstractmethod
    def __iter__(self) -> Iterator[MissionFrame]:
        """Yield MissionFrame objects in capture order."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames (may be 0 for live streams)."""
        ...

    @property
    @abstractmethod
    def mission_id(self) -> str:
        """Unique mission identifier."""
        ...
