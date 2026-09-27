"""
mission/recorded.py — Recorded mission source.

Iterates over sonar frames saved on disk from a completed AUV or
vessel mission. This is the primary data source for the MVP.
"""
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional

from inference.types import MissionFrame
from mission.source import MissionSource

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",".xtf"}


class RecordedMissionSource(MissionSource):
    """
    Iterate over sonar frames from a recorded mission directory.

    Expected directory layout:
        mission_dir/
            sonar/
                frame_000001.png
                frame_000002.png
                ...
            navigation.csv       (optional)
            metadata.json        (optional)
            mission.json         (optional)

    Args:
        mission_dir: Path to the mission directory.
        sonar_subdir: Subdirectory containing sonar images (default: "sonar").
    """

    def __init__(
        self,
        mission_dir: str,
        sonar_subdir: str = "sonar",
    ):
        self._mission_dir = Path(mission_dir).expanduser().resolve()
        self._sonar_dir = self._mission_dir / sonar_subdir
        self._mission_meta = self._load_mission_meta()
        self._frames = self._discover_frames()

        logger.info(
            "RecordedMissionSource: mission=%s, frames=%d",
            self.mission_id, len(self._frames),
        )

    @property
    def mission_id(self) -> str:
        return self._mission_meta.get("mission_id", self._mission_dir.name)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[MissionFrame]:
        for idx, path in enumerate(self._frames):
            yield MissionFrame(
                frame_id=path.stem,
                image_path=str(path),
                mission_id=self.mission_id,
                frame_index=idx,
                timestamp=self._mission_meta.get("timestamps", {}).get(path.stem),
            )

    @property
    def navigation_file(self) -> Optional[str]:
        """Path to navigation CSV if it exists."""
        for name in ["navigation.csv", "nav.csv", "gps.csv"]:
            p = self._mission_dir / name
            if p.exists():
                return str(p)
        return None

    def _discover_frames(self) -> List[Path]:
        """Find and sort sonar image files."""
        if not self._sonar_dir.exists():
            # Fall back to images directly in mission dir
            candidates = [
                p for p in sorted(self._mission_dir.iterdir())
                if p.suffix.lower() in IMAGE_EXTS
            ]
            if candidates:
                logger.info("No sonar/ subdir, using %d images from root", len(candidates))
                return candidates
            logger.warning("No sonar images found in %s", self._mission_dir)
            return []

        frames = sorted(
            p for p in self._sonar_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        return frames

    def _load_mission_meta(self) -> dict:
        """Load mission.json or metadata.json if present."""
        for name in ["mission.json", "metadata.json"]:
            p = self._mission_dir / name
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load %s: %s", p, e)
        return {}
