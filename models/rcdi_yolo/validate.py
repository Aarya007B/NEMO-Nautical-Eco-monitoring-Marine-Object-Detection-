"""
models/rcdi_yolo/validate.py

RCDI-YOLO validation entry point.

Usage:
    python -m models.rcdi_yolo.validate --config configs/config.yaml --weights models/rcdi_yolo/weights/rcdi_yolo_1c_best.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


def validate_model(config_path: str, weights_path: str):
    """Run validation on a trained RCDI-YOLO model."""
    from models.rcdi_yolo.model import RCDIYOLOModel
    from models.rcdi_yolo.parser import get_model_kwargs_from_file
    from models.rcdi_yolo.dataset import SonarDetectionDataset
    from torch.utils.data import DataLoader

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    model_kwargs = get_model_kwargs_from_file(config_path.replace("config.yaml", "rcdi_yolo_1c.yaml") if Path(config_path).name == "config.yaml" else config_path)
    if Path("configs/rcdi_yolo_1c.yaml").exists():
        from models.rcdi_yolo.parser import parse_rcdi_config
        model_kwargs = parse_rcdi_config(yaml.safe_load(Path("configs/rcdi_yolo_1c.yaml").read_text()))

    model = RCDIYOLOModel(**model_kwargs).to(device)

    if weights_path and Path(weights_path).exists():
        model.load_weights(weights_path)
        logger.info("Loaded weights: %s", weights_path)
    else:
        logger.warning("No weights found at %s", weights_path)
        return

    model.eval()

    # Build validation dataset
    splits_dir = cfg.get("data", {}).get("splits_dir", "data/splits")
    val_file = Path(splits_dir) / "val.txt"
    if not val_file.exists():
        logger.warning("Validation split not found: %s", val_file)
        return

    with open(val_file) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if not image_paths:
        logger.warning("No validation images found.")
        return

    dataset = SonarDetectionDataset(
        image_paths=image_paths,
        annotation_paths=[],
        image_size=cfg["input"].get("image_size", 640),
        num_classes=model_kwargs["num_classes"],
        augment=False,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Validation metrics
    total_detections = 0
    total_targets = 0
    total_iou = 0.0
    num_valid = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch or not batch[0]:
                continue
            img_tensor = batch[0].to(device)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            predictions = model(img_tensor)
            dets = model.predict(img_tensor, conf_threshold=0.5)
            total_detections += len(dets)

            if batch_idx < 5:
                logger.info(
                    "Sample frame %d: %d detections",
                    batch_idx, len(dets),
                )

    logger.info("=" * 55)
    logger.info("RCDI-YOLO Validation Results")
    logger.info("=" * 55)
    logger.info("  Total detections: %d", total_detections)
    logger.info("  Total targets:    %d", total_targets)
    logger.info("  Devices:          %s", device)
    logger.info("=" * 55)
    logger.info(
        "NOTE: Full mAP@0.5, Precision, Recall, F1 require annotated datasets. "
        "Add a proper evaluation module for production use."
    )


def main():
    parser = argparse.ArgumentParser(description="RCDI-YOLO Validator")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", default="models/rcdi_yolo/weights/rcdi_yolo_1c_best.pt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    validate_model(args.config, args.weights)


if __name__ == "__main__":
    main()
