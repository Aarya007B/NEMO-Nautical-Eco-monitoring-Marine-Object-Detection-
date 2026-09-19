"""
reporting/csv_export.py — CSV export for detection results.
"""
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from inference.types import DetectionResult, MissionResult

logger = logging.getLogger(__name__)


class CSVExporter:
    """Exports detection results to CSV format."""

    COLUMNS = [
        "detection_id", "frame_id", "mission_id",
        "x1", "y1", "x2", "y2",
        "detector_confidence", "detector_class",
        "verifier_class", "artificial_probability", "natural_probability",
        "artificialness_score", "priority_score",
        "latitude", "longitude", "timestamp",
        "status", "crop_path",
    ]

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_mission(
        self,
        results: List[MissionResult],
        mission_id: str = "unknown",
    ) -> str:
        """
        Export full mission results to CSV.

        Returns:
            Path to the saved CSV file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mission_{mission_id}_{timestamp}.csv"
        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            for result in results:
                for det in result.detections:
                    writer.writerow({
                        "detection_id": det.detection_id,
                        "frame_id": result.frame_id,
                        "mission_id": result.mission_id,
                        "x1": det.bbox.x1,
                        "y1": det.bbox.y1,
                        "x2": det.bbox.x2,
                        "y2": det.bbox.y2,
                        "detector_confidence": round(det.detector_confidence, 4),
                        "detector_class": det.detector_class,
                        "verifier_class": det.verifier_class,
                        "artificial_probability": round(det.artificial_probability, 4),
                        "natural_probability": round(det.natural_probability, 4),
                        "artificialness_score": round(det.artificialness_score, 4),
                        "priority_score": round(det.priority_score, 4),
                        "latitude": det.latitude,
                        "longitude": det.longitude,
                        "timestamp": det.timestamp,
                        "status": det.status.value if hasattr(det.status, "value") else str(det.status),
                        "crop_path": det.crop_path,
                    })

        total_dets = sum(len(r.detections) for r in results)
        logger.info("CSV export: %s (%d detections)", filepath, total_dets)
        return str(filepath)
