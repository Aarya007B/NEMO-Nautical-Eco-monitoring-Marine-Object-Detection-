"""
metadata/ping.py — Sonar ping metadata parser.

Reads ping-level metadata from sonar log files.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from metadata.schema import PingMetadata

logger = logging.getLogger(__name__)


class PingParser:
    """
    Parses ping metadata from JSON or sonar log files.

    Supports:
        - JSON array of ping records
        - JSON lines format
    """

    def parse(self, filepath: str) -> List[PingMetadata]:
        """
        Parse a ping metadata file.

        Args:
            filepath: Path to JSON ping log.

        Returns:
            List of PingMetadata objects.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("Ping metadata file not found: %s", filepath)
            return []

        try:
            text = path.read_text()
            data = json.loads(text)
            if isinstance(data, list):
                return [self._parse_record(r) for r in data]
            elif isinstance(data, dict) and "pings" in data:
                return [self._parse_record(r) for r in data["pings"]]
        except json.JSONDecodeError:
            # Try JSON lines
            records = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(self._parse_record(json.loads(line)))
                except json.JSONDecodeError:
                    continue
            return records

        logger.warning("Could not parse ping metadata: %s", filepath)
        return []

    def _parse_record(self, rec: dict) -> PingMetadata:
        return PingMetadata(
            ping_number=int(rec.get("ping_number", rec.get("ping", 0))),
            timestamp=float(rec.get("timestamp", 0.0)),
            range_m=float(rec.get("range_m", rec.get("range", 0.0))),
            frequency_khz=float(rec.get("frequency_khz", rec.get("frequency", 0.0))),
            gain_db=float(rec.get("gain_db", rec.get("gain", 0.0))),
            slant_range_correction=bool(rec.get("slant_range_correction", False)),
        )

    def to_dict(self, pings: List[PingMetadata]) -> Dict[int, PingMetadata]:
        """Index pings by ping_number for fast lookup."""
        return {p.ping_number: p for p in pings}
