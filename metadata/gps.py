"""
metadata/gps.py — GPS/navigation log parser.

Reads CSV navigation logs commonly exported by AUV or vessel navigation
systems and converts them to NavigationRecord objects.
"""
import csv
import logging
from pathlib import Path
from typing import List, Optional

from metadata.schema import NavigationRecord

logger = logging.getLogger(__name__)


class GPSParser:
    """
    Parses navigation CSV files into NavigationRecord objects.

    Supports common column names:
        timestamp, latitude/lat, longitude/lon/lng, heading, depth, altitude, speed
    """

    COLUMN_ALIASES = {
        "timestamp": ["timestamp", "time", "epoch", "t"],
        "latitude":  ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng", "long"],
        "heading":   ["heading", "hdg", "yaw"],
        "depth":     ["depth", "z"],
        "altitude":  ["altitude", "alt"],
        "speed":     ["speed", "velocity", "v"],
    }

    def parse(self, filepath: str) -> List[NavigationRecord]:
        """
        Parse a navigation CSV file.

        Args:
            filepath: Path to CSV file.

        Returns:
            List of NavigationRecord objects.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("Navigation file not found: %s", filepath)
            return []

        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.warning("Empty CSV: %s", filepath)
                return []

            col_map = self._resolve_columns(reader.fieldnames)

            for row in reader:
                try:
                    rec = NavigationRecord(
                        timestamp=self._get_float(row, col_map.get("timestamp")),
                        latitude=self._get_float(row, col_map.get("latitude")),
                        longitude=self._get_float(row, col_map.get("longitude")),
                        heading=self._get_float(row, col_map.get("heading")),
                        depth=self._get_float(row, col_map.get("depth")),
                        altitude=self._get_float(row, col_map.get("altitude")),
                        speed=self._get_float(row, col_map.get("speed")),
                    )
                    records.append(rec)
                except (ValueError, KeyError) as e:
                    logger.debug("Skipping row: %s", e)

        logger.info("Parsed %d navigation records from %s", len(records), filepath)
        return records

    def _resolve_columns(self, fieldnames: list) -> dict:
        col_map = {}
        lower_fields = {f.lower().strip(): f for f in fieldnames}
        for key, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_fields:
                    col_map[key] = lower_fields[alias]
                    break
        return col_map

    @staticmethod
    def _get_float(row: dict, col: Optional[str]) -> float:
        if col is None or col not in row or row[col] == "":
            return 0.0
        return float(row[col])
