"""data_adapters/dfki_uxo.py — DFKI-RIC UXO Dataset 2024 adapter."""
import logging
from pathlib import Path
from typing import List

from data_adapters.base import BaseDatasetAdapter
from inference.types import DatasetRecord

logger = logging.getLogger(__name__)


class DFKIUXOAdapter(BaseDatasetAdapter):

    def load(self) -> List[DatasetRecord]:
        records = []
        images_dir = Path(self.root) / "images"

        if not images_dir.exists():
            logger.warning("DFKI UXO images dir not found: %s", images_dir)
            return records

        for img_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
            ann_path = Path(self.root) / "annotations" / (img_path.stem + ".txt")
            records.append(DatasetRecord(
                image_path=str(img_path),
                annotation_path=str(ann_path) if ann_path.exists() else None,
                source_dataset="dfki_uxo",
                source_class="UXO",
                canonical_class=self.CANONICAL_CLASS,
                frame_id=img_path.stem,
            ))

        logger.info("DFKI UXO: loaded %d records", len(records))
        return self.validate(records)
