"""
data_adapters/base.py

Base class for all NEMO dataset adapters.
Every adapter converts its source dataset into DatasetRecord objects.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List

from inference.types import DatasetRecord

logger = logging.getLogger(__name__)


class BaseDatasetAdapter(ABC):
    """
    Abstract base class for NEMO dataset adapters.

    Args:
        root (str): Path to the raw dataset root directory.
        canonical_class (str): Canonical class name for Stage-1 detector.
    """

    CANONICAL_CLASS = "target"

    def __init__(self, root: str, canonical_class: str = "target"):
        self.root = root
        self.canonical_class = canonical_class
        logger.info("%s: root=%s", self.__class__.__name__, root)

    @abstractmethod
    def load(self) -> List[DatasetRecord]:
        """Load dataset and return a list of DatasetRecord objects."""
        ...

    def validate(self, records: List[DatasetRecord]) -> List[DatasetRecord]:
        """Filter out records with missing images."""
        valid = []
        for r in records:
            if not r.image_path or not os.path.exists(r.image_path):
                logger.warning("Image not found, skipping: %s", r.image_path)
            else:
                valid.append(r)
        logger.info(
            "%s: %d/%d records valid",
            self.__class__.__name__, len(valid), len(records),
        )
        return valid
