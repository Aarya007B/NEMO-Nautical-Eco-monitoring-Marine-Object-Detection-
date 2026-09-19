"""
data_adapters/uci_sonar.py — UCI Sonar Mines vs Rocks adapter.

Note: This dataset contains feature vectors, NOT images.
"""
import csv
import logging
from pathlib import Path
from typing import List

from data_adapters.base import BaseDatasetAdapter
from inference.types import DatasetRecord

logger = logging.getLogger(__name__)


class UCISonarAdapter(BaseDatasetAdapter):

    def load(self) -> List[DatasetRecord]:
        data_file = Path(self.root) / "sonar.all-data"
        if not data_file.exists():
            logger.warning("UCI Sonar data file not found: %s", data_file)
            return []

        records = []
        with open(data_file, newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) < 2:
                    continue
                label = row[-1].strip().upper()
                source_class = "Mine" if label == "M" else "Rock"
                canonical = "target" if label == "M" else "natural"
                records.append(DatasetRecord(
                    image_path="",
                    annotation_path=None,
                    source_dataset="uci_sonar",
                    source_class=source_class,
                    canonical_class=canonical,
                    frame_id=f"uci_{i:05d}",
                    metadata={"features": [float(v) for v in row[:-1]]},
                ))

        logger.info("UCI Sonar: loaded %d records (feature vectors)", len(records))
        return records
