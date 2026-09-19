"""
models/mobilenetv3/validate.py

MobileNetV3-Small verifier validation entry point.

Usage:
    python -m models.mobilenetv3.validate --config configs/config.yaml --weights models/mobilenetv3/weights/mobilenetv3_verifier_best.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


def validate():
    """Validate the MobileNetV3 verifier."""
    parser = argparse.ArgumentParser(description="MobileNetV3 Verifier Validator")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", default="models/mobilenetv3/weights/mobilenetv3_verifier_best.pt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    from models.mobilenetv3.model import MobileNetV3Verifier
    from models.mobilenetv3.dataset import CropDataset
    from torch.utils.data import DataLoader

    verifier_cfg = cfg.get("verifier", {})
    model = MobileNetV3Verifier(
        in_channels=verifier_cfg.get("input_channels", 1),
        num_classes=verifier_cfg.get("num_classes", 2),
        conf_threshold=verifier_cfg.get("confidence_threshold", 0.5),
    ).to(device)

    if args.weights and Path(args.weights).exists():
        state = torch.load(args.weights, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        else:
            model.load_weights(args.weights)
        logger.info("Loaded weights: %s", args.weights)
    else:
        logger.warning("No weights found at %s", args.weights)
        return

    model.eval()

    dataset_cfg = verifier_cfg.get("dataset", {})
    root = dataset_cfg.get("root", "data/hard_negatives")

    val_dataset = CropDataset(
        root=Path(root) / "val",
        crop_size=verifier_cfg.get("crop_size", 128),
    )

    if len(val_dataset) == 0:
        logger.warning("No validation data found. Using synthetic test...")
        fake_input = torch.randn(4, 1, 128, 128).to(device)
        logits = model(fake_input)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        logger.info("Synthetic test: logits=%s, preds=%s", logits.shape, preds.shape)
        logger.info("Natural prob range: [%.3f, %.3f]", probs[:, 0].min(), probs[:, 0].max())
        logger.info("Artificial prob range: [%.3f, %.3f]", probs[:, 1].min(), probs[:, 1].max())
        return

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    natural_correct = 0
    artificial_correct = 0
    natural_total = 0
    artificial_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            if len(images) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            natural_mask = labels == 0
            artificial_mask = labels == 1
            natural_correct += preds[natural_mask].eq(labels[natural_mask]).sum().item()
            artificial_correct += preds[artificial_mask].eq(labels[artificial_mask]).sum().item()
            natural_total += natural_mask.sum().item()
            artificial_total += artificial_mask.sum().item()

    acc = correct / max(total, 1)
    natural_acc = natural_correct / max(natural_total, 1)
    artificial_acc = artificial_correct / max(artificial_total, 1)

    logger.info("=" * 55)
    logger.info("MobileNetV3 Verifier Validation Results")
    logger.info("=" * 55)
    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  Natural:   %.4f (%d samples)", natural_acc, natural_total)
    logger.info("  Artificial:%.4f (%d samples)", artificial_acc, artificial_total)
    logger.info("=" * 55)


def main():
    validate()


if __name__ == "__main__":
    main()
