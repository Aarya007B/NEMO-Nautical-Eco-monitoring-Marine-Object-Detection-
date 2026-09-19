"""
tools/validate_dataset.py — Validate dataset structure and annotations.

Usage:
    python tools/validate_dataset.py [--root data/raw]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_directory(root: str):
    root_path = Path(root)
    if not root_path.exists():
        logger.error("Directory not found: %s", root)
        return

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = [p for p in root_path.rglob("*") if p.suffix.lower() in image_exts]
    labels = list(root_path.rglob("*.txt"))
    jsons = list(root_path.rglob("*.json"))

    logger.info("Dataset root: %s", root_path.absolute())
    logger.info("  Images found: %d", len(images))
    logger.info("  Label files (.txt): %d", len(labels))
    logger.info("  JSON files: %d", len(jsons))

    # Check for common issues
    if not images:
        logger.warning("  No images found!")

    # Check image-label pairing
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    unpaired_images = image_stems - label_stems
    unpaired_labels = label_stems - image_stems

    if unpaired_images:
        logger.warning("  Images without labels: %d", len(unpaired_images))
    if unpaired_labels:
        logger.warning("  Labels without images: %d", len(unpaired_labels))

    paired = image_stems & label_stems
    logger.info("  Paired image-label: %d", len(paired))


def main():
    parser = argparse.ArgumentParser(description="Validate Dataset")
    parser.add_argument("--root", default="data/raw", help="Dataset root directory")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        logger.error("Root not found: %s", root)
        return

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs:
        for d in sorted(subdirs):
            logger.info("\n--- %s ---", d.name)
            validate_directory(str(d))
    else:
        validate_directory(str(root))


if __name__ == "__main__":
    main()
