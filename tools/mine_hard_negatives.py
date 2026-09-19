"""
tools/mine_hard_negatives.py — Hard negative mining for verifier training.

Runs the detector on validation/negative data, collects false positive
crops, and saves them for verifier retraining.

Usage:
    python tools/mine_hard_negatives.py --config configs/config.yaml --data data/raw/negatives
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mine Hard Negatives")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", required=True, help="Path to negative/seabed images")
    parser.add_argument("--output", default="data/hard_negatives", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.10, help="Low confidence threshold")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Hard Negative Mining")
    logger.info("  Data source: %s", args.data)
    logger.info("  Output: %s", args.output)
    logger.info("  Confidence threshold: %.2f", args.conf)

    # Import detector
    import yaml
    import cv2
    import numpy as np

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from inference.detector import CandidateDetector

    detector = CandidateDetector(
        detector_config={**config.get("detector", {}), "confidence_threshold": args.conf},
        preprocessing_config=config.get("preprocessing", {}),
    )

    # Find images
    data_dir = Path(args.data)
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    images = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in image_exts)
    logger.info("  Found %d images", len(images))

    mined = 0
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detections, _, scale, padding = detector.detect(img, frame_id=img_path.stem)

        for i, det in enumerate(detections):
            # Extract crop
            h, w = img.shape[:2]
            expanded = det.bbox.expand(0.15, w, h)
            x1, y1 = max(0, int(expanded.x1)), max(0, int(expanded.y1))
            x2, y2 = min(w, int(expanded.x2)), min(h, int(expanded.y2))

            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (128, 128))
                crop_name = f"{img_path.stem}_fp_{i:03d}.png"
                cv2.imwrite(str(output_dir / crop_name), crop)
                mined += 1

    logger.info("  Mined %d hard negative crops -> %s", mined, args.output)
    logger.info("  Review and relabel before adding to verifier training data.")


if __name__ == "__main__":
    main()
