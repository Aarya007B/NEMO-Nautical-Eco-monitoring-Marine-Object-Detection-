"""data_adapters/opensonardatasets.py — REMARO OpenSonarDatasets adapter."""
import logging
from pathlib import Path
from typing import List

from data_adapters.base import BaseDatasetAdapter
from inference.types import DatasetRecord

logger = logging.getLogger(__name__)


class OpenSonarDatasetsAdapter(BaseDatasetAdapter):

    def load(self) -> List[DatasetRecord]:
        records = []
        root = Path(self.root)
        for img_path in sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg")):
            ann = img_path.with_suffix(".txt")
            records.append(DatasetRecord(
                image_path=str(img_path),
                annotation_path=str(ann) if ann.exists() else None,
                source_dataset="opensonardatasets",
                source_class="sonar_object",
                canonical_class=self.CANONICAL_CLASS,
                frame_id=img_path.stem,
                mission_id=img_path.parent.name,
            ))
        logger.info("OpenSonarDatasets: loaded %d records", len(records))
        return self.validate(records)
