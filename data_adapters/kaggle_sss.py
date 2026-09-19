"""data_adapters/kaggle_sss.py — Kaggle Side-Scan Sonar adapter."""
import logging
from pathlib import Path
from typing import List

from data_adapters.base import BaseDatasetAdapter
from inference.types import DatasetRecord

logger = logging.getLogger(__name__)


class KaggleSSSAdapter(BaseDatasetAdapter):

    def load(self) -> List[DatasetRecord]:
        records = []
        images_dir = Path(self.root) / "images"
        if not images_dir.exists():
            logger.warning("Kaggle SSS images dir not found: %s", images_dir)
            return records

        for img_path in sorted(images_dir.rglob("*.png")) + sorted(images_dir.rglob("*.jpg")):
            ann = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            records.append(DatasetRecord(
                image_path=str(img_path),
                annotation_path=str(ann) if ann.exists() else None,
                source_dataset="kaggle_sss",
                source_class="sonar_target",
                canonical_class=self.CANONICAL_CLASS,
                frame_id=img_path.stem,
            ))
        logger.info("Kaggle SSS: loaded %d records", len(records))
        return self.validate(records)
