"""
reporting/report_generator.py — Unified report generation.

Orchestrates JSON + CSV export and generates a summary report.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from inference.types import MissionResult
from reporting.csv_export import CSVExporter
from reporting.json_export import JSONExporter

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates mission reports in multiple formats.

    Args:
        output_config: Output configuration from config.yaml.
    """

    def __init__(self, output_config: Dict):
        output_dir = output_config.get("directory", "outputs/")
        reports_dir = output_dir.rstrip("/") + "/reports"

        self.json_exporter = JSONExporter(reports_dir)
        self.csv_exporter = CSVExporter(reports_dir)
        self.save_json = output_config.get("save_json", True)
        self.save_csv = output_config.get("save_csv", True)

    def generate(
        self,
        results: List[MissionResult],
        mission_id: str = "unknown",
    ) -> Dict[str, str]:
        """
        Generate all configured reports.

        Returns:
            Dict mapping format name to file path.
        """
        outputs = {}

        if self.save_json:
            path = self.json_exporter.export_mission(results, mission_id)
            outputs["json"] = path

        if self.save_csv:
            path = self.csv_exporter.export_mission(results, mission_id)
            outputs["csv"] = path

        # Print summary
        total_frames = len(results)
        total_dets = sum(len(r.detections) for r in results)
        verified = sum(
            1 for r in results for d in r.detections
            if hasattr(d.status, "value") and d.status.value == "verified"
        )

        logger.info(
            "Report: mission=%s, frames=%d, detections=%d, verified=%d",
            mission_id, total_frames, total_dets, verified,
        )

        return outputs
