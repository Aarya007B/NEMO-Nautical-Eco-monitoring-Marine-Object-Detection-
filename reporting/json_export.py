"""
reporting/json_export.py — JSON export for detection results.
"""
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

from inference.types import DetectionResult, MissionResult

logger = logging.getLogger(__name__)


class JSONExporter:
    """Exports detection results to JSON format."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_mission(
        self,
        results: List[MissionResult],
        mission_id: str = "unknown",
    ) -> str:
        """
        Export full mission results to JSON.

        Returns:
            Path to the saved JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mission_{mission_id}_{timestamp}.json"
        filepath = self.output_dir / filename

        export_data = {
            "mission_id": mission_id,
            "export_timestamp": datetime.now().isoformat(),
            "total_frames": len(results),
            "total_detections": sum(len(r.detections) for r in results),
            "frames": [],
        }

        for result in results:
            frame_data = {
                "frame_id": result.frame_id,
                "detections": [],
            }
            for det in result.detections:
                det_dict = {
                    "detection_id": det.detection_id,
                    "bbox": {
                        "x1": det.bbox.x1,
                        "y1": det.bbox.y1,
                        "x2": det.bbox.x2,
                        "y2": det.bbox.y2,
                    },
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
                }
                frame_data["detections"].append(det_dict)
            export_data["frames"].append(frame_data)

        filepath.write_text(json.dumps(export_data, indent=2, default=str))
        logger.info("JSON export: %s (%d frames)", filepath, len(results))
        return str(filepath)

    def export_detections(
        self,
        detections: List[DetectionResult],
        filename: str = "detections.json",
    ) -> str:
        """Export a flat list of detections."""
        filepath = self.output_dir / filename
        data = []
        for det in detections:
            data.append({
                "detection_id": det.detection_id,
                "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                "detector_confidence": round(det.detector_confidence, 4),
                "verifier_class": det.verifier_class,
                "artificial_probability": round(det.artificial_probability, 4),
                "artificialness_score": round(det.artificialness_score, 4),
                "priority_score": round(det.priority_score, 4),
                "status": det.status.value if hasattr(det.status, "value") else str(det.status),
                "latitude": det.latitude,
                "longitude": det.longitude,
            })
        filepath.write_text(json.dumps(data, indent=2, default=str))
        logger.info("JSON detection export: %s (%d detections)", filepath, len(data))
        return str(filepath)
