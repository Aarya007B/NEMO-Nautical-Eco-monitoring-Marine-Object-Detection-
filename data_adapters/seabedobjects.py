"""
data_adapters/seabedobjects.py

Adapter for the SeabedObjects Ship-and-Airplane dataset.
Supports both flat directory structure (images/, labels/) and
Roboflow YOLO format (train/, valid/, test/ with data.yaml).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from data_adapters.base import BaseDatasetAdapter
from inference.types import DatasetRecord

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SeabedObjectsAdapter(BaseDatasetAdapter):
    """
    Adapter for SeabedObjects Ship-and-Airplane dataset.
    Maps source classes (ship, airplane) to canonical 'target'.
    """

    def load(self) -> List[DatasetRecord]:
        records = []
        root_path = Path(self.root)

        if not root_path.exists():
            logger.warning("SeabedObjects root not found: %s", self.root)
            return records

        # 1. Parse data.yaml if present (Roboflow / Ultralytics export)
        class_names = self._parse_data_yaml(root_path / "data.yaml")

        # 2. Check for Roboflow split directories (train, valid/val, test)
        splits = ["train", "valid", "val", "test"]
        split_dirs = [root_path / s for s in splits if (root_path / s).is_dir()]

        if split_dirs:
            logger.info("Found %d split directory(ies) in %s", len(split_dirs), self.root)
            for s_dir in split_dirs:
                split_name = "val" if s_dir.name == "valid" else s_dir.name
                # Roboflow structure: train/images and train/labels
                imgs_dir = s_dir / "images" if (s_dir / "images").is_dir() else s_dir
                lbls_dir = s_dir / "labels" if (s_dir / "labels").is_dir() else s_dir

                for img_path in sorted(imgs_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTS:
                        label_path = lbls_dir / f"{img_path.stem}.txt"
                        records.append(DatasetRecord(
                            image_path=str(img_path),
                            annotation_path=str(label_path) if label_path.exists() else None,
                            source_dataset="seabedobjects",
                            source_class="ship_or_airplane",
                            canonical_class=self.canonical_class,
                            frame_id=img_path.stem,
                            metadata={"split": split_name, "class_names": class_names},
                        ))

        # 3. Fallback: Flat images/ and labels/ directories
        else:
            images_dir = root_path / "images" if (root_path / "images").is_dir() else root_path
            labels_dir = root_path / "labels" if (root_path / "labels").is_dir() else root_path

            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTS:
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    records.append(DatasetRecord(
                        image_path=str(img_path),
                        annotation_path=str(label_path) if label_path.exists() else None,
                        source_dataset="seabedobjects",
                        source_class="ship_or_airplane",
                        canonical_class=self.canonical_class,
                        frame_id=img_path.stem,
                        metadata={"class_names": class_names},
                    ))

        logger.info("SeabedObjects: loaded %d records", len(records))
        return self.validate(records)

    def _parse_data_yaml(self, yaml_path: Path) -> Dict[int, str]:
        """Parse class names from data.yaml if available."""
        if not yaml_path.exists():
            return {0: "ship", 1: "airplane"}

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, list):
                    return {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
        except Exception as e:
            logger.warning("Could not parse %s: %s", yaml_path, e)

        return {0: "ship", 1: "airplane"}
