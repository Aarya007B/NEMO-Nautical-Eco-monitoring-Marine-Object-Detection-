"""
mission/video.py — Video-file mission source.

Extracts frames from a recorded mission video (MP4, AVI, MOV, MKV, …)
and feeds them into the standard MissionPipeline as MissionFrame objects.

Key decisions
-------------
- Frames are saved to disk in a temporary/output directory so that the
  existing pipeline (which expects image_path strings) works unchanged.
- Frame sampling is configurable: process every N-th frame or sample at
  a fixed FPS to avoid flooding the pipeline with near-identical sonar
  pings.
- The source uses OpenCV VideoCapture, which supports all common video
  container formats that OpenCV was built with (H264/MP4, MJPEG/AVI,
  VP9/WebM, etc.).

Usage
-----
    source = VideoMissionSource(
        video_path="mission_footage.mp4",
        output_dir="outputs/frames/",      # where extracted frames are saved
        frame_interval=5,                  # process every 5th frame
        mission_id="MISSION_2024_001",     # optional override
    )
    pipeline = MissionPipeline.from_config(config)
    results   = pipeline.process_mission(source)
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Iterator, Optional

import cv2

from inference.types import MissionFrame
from mission.source import MissionSource

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


class VideoMissionSource(MissionSource):
    """
    Iterate over frames extracted from a mission video file.

    The video is read with ``cv2.VideoCapture``.  Frames are saved to
    *output_dir* as grayscale PNG files so the unchanged pipeline can
    load them via ``cv2.imread``.

    Args:
        video_path:     Path to the video file.
        output_dir:     Directory to write extracted frames into.
                        Created automatically if it does not exist.
        frame_interval: Extract every N-th frame (default 1 = every frame).
                        Use higher values for high-FPS video where
                        consecutive frames are nearly identical sonar pings.
        target_fps:     Alternative to frame_interval — keep approximately
                        this many frames per second of video.  Overrides
                        frame_interval when set.
        mission_id:     Optional mission ID override.  Defaults to the
                        video file stem.
        save_frames:    If False, yield frames from memory without writing
                        to disk (not recommended — kept for future use).
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str = "outputs/video_frames",
        frame_interval: int = 1,
        target_fps: Optional[float] = None,
        mission_id: Optional[str] = None,
        save_frames: bool = True,
    ):
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if self._video_path.suffix.lower() not in VIDEO_EXTS:
            logger.warning(
                "Unusual video extension '%s'. Proceeding anyway.",
                self._video_path.suffix,
            )

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._save_frames = save_frames
        self._mission_id = mission_id or self._video_path.stem

        # Open video to read metadata
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        self._total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Resolve frame_interval from target_fps if given
        if target_fps is not None and target_fps > 0:
            self._frame_interval = max(1, round(self._source_fps / target_fps))
        else:
            self._frame_interval = max(1, int(frame_interval))

        # Pre-compute which frame indices will be extracted
        self._frame_indices = list(
            range(0, self._total_video_frames, self._frame_interval)
        )

        logger.info(
            "VideoMissionSource: mission=%s | video=%s | "
            "source_fps=%.1f | interval=%d | frames_to_process=%d",
            self.mission_id,
            self._video_path.name,
            self._source_fps,
            self._frame_interval,
            len(self._frame_indices),
        )

    # ------------------------------------------------------------------
    # MissionSource interface
    # ------------------------------------------------------------------

    @property
    def mission_id(self) -> str:
        return self._mission_id

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __iter__(self) -> Iterator[MissionFrame]:
        """Open the video once and yield MissionFrame objects in order."""
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            logger.error("VideoCapture failed to open: %s", self._video_path)
            return

        current_video_idx = 0
        extracted_count = 0

        try:
            for target_idx in self._frame_indices:
                # Seek forward to the target frame index efficiently
                if target_idx != current_video_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    current_video_idx = target_idx

                ret, bgr_frame = cap.read()
                if not ret:
                    logger.warning(
                        "Frame %d could not be read — stopping early.", target_idx
                    )
                    break

                current_video_idx += 1

                # Convert to grayscale (NEMO pipeline is 1-channel sonar)
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

                # Build a stable frame ID
                frame_id = f"{self._mission_id}_f{target_idx:06d}"

                # Compute approximate timestamp in seconds
                timestamp_sec = target_idx / self._source_fps

                if self._save_frames:
                    frame_path = self._output_dir / f"{frame_id}.png"
                    cv2.imwrite(str(frame_path), gray)
                    image_path = str(frame_path)
                else:
                    # Fallback: write to a temp file (pipeline needs a path)
                    tmp_path = self._output_dir / f"_tmp_{uuid.uuid4().hex[:8]}.png"
                    cv2.imwrite(str(tmp_path), gray)
                    image_path = str(tmp_path)

                extracted_count += 1
                yield MissionFrame(
                    mission_id=self._mission_id,
                    frame_id=frame_id,
                    image_path=image_path,
                    frame_index=extracted_count - 1,
                    timestamp=timestamp_sec,
                )

        finally:
            cap.release()
            logger.info(
                "VideoMissionSource done: %d frames extracted from %s",
                extracted_count,
                self._video_path.name,
            )

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    @property
    def video_path(self) -> str:
        return str(self._video_path)

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def frame_interval(self) -> int:
        return self._frame_interval

    @property
    def frames_dir(self) -> str:
        """Directory where extracted frames are written."""
        return str(self._output_dir)

    @property
    def video_resolution(self) -> tuple:
        return (self._width, self._height)
