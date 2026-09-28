"""
models/mobilenetv3/train.py

MobileNetV3-Small verifier training entry point.

Usage:
    python -m models.mobilenetv3.train --config configs/config.yaml --epochs 10
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


def train():
    """Train the MobileNetV3 verifier."""
    parser = argparse.ArgumentParser(description="MobileNetV3 Verifier Trainer")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None, help="Root directory for dataset crops (e.g. data/raw/marine_pulse)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default="models/mobilenetv3/weights")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("NEMO MobileNetV3 Verifier Trainer")
    logger.info("=" * 60)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_pref = cfg.get("runtime", {}).get("device", "auto")
    if device_pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_pref)
    logger.info("Device: %s", device)

    from models.mobilenetv3.model import MobileNetV3Verifier
    from models.mobilenetv3.dataset import CropDataset
    from torch.utils.data import DataLoader

    verifier_cfg = cfg.get("verifier", cfg)
    if "config" in verifier_cfg and Path(verifier_cfg["config"]).exists():
        with open(verifier_cfg["config"]) as f:
            sub_cfg = yaml.safe_load(f) or {}
            # Merge sub-config values
            for k, v in sub_cfg.items():
                if k not in verifier_cfg:
                    verifier_cfg[k] = v

    crop_size = verifier_cfg.get("crop_size") or verifier_cfg.get("model", {}).get("image_size", 128)
    in_channels = verifier_cfg.get("input_channels") or verifier_cfg.get("model", {}).get("input_channels", 1)
    num_classes = verifier_cfg.get("num_classes") or verifier_cfg.get("model", {}).get("num_classes", 2)
    conf_threshold = verifier_cfg.get("confidence_threshold", 0.5)

    model = MobileNetV3Verifier(
        in_channels=in_channels,
        num_classes=num_classes,
        conf_threshold=conf_threshold,
    ).to(device)
    logger.info("Model: %.2fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=verifier_cfg.get("training", {}).get("learning_rate", 0.001),
        weight_decay=verifier_cfg.get("training", {}).get("weight_decay", 0.0001),
    )

    epochs = args.epochs or verifier_cfg.get("training", {}).get("epochs", 50)
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # Build dataset
    dataset_cfg = verifier_cfg.get("dataset", {})
    default_root = "data/raw/opensonardatasets/Marine_PULSE 2" if Path("data/raw/opensonardatasets/Marine_PULSE 2").exists() else (
        "data/raw/marine_pulse" if Path("data/raw/marine_pulse").exists() else "data/hard_negatives"
    )
    root = args.data or dataset_cfg.get("root", default_root)

    root_path = Path(root)
    if (root_path / "train").exists():
        train_dataset = CropDataset(root=root_path / "train", crop_size=crop_size)
        val_dir = root_path / "val" if (root_path / "val").exists() else root_path / "test"
        if val_dir.exists():
            val_dataset = CropDataset(root=val_dir, crop_size=crop_size)
        else:
            val_size = max(1, int(len(train_dataset) * 0.15))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
    else:
        full_dataset = CropDataset(root=root_path, crop_size=crop_size)
        if len(full_dataset) > 0:
            val_size = max(1, int(len(full_dataset) * 0.15))
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            logger.info("Split %s into %d train, %d val", root, train_size, val_size)
        else:
            train_dataset, val_dataset = [], []

    if len(train_dataset) == 0:
        logger.warning(
            "No training data found at %s", root)
        logger.warning("Run hard negative mining first or add training crops.")
        logger.warning("Falling back to synthetic training...")
        for epoch in range(min(3, epochs)):
            fake_input = torch.randn(4, 1, 128, 128).to(device)
            fake_target = torch.randint(0, 2, (4,)).to(device)
            logits = model(fake_input)
            loss = criterion(logits, fake_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info("Epoch %d: synthetic loss=%.4f", epoch, loss.item())
        logger.info("Synthetic training complete.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=verifier_cfg.get("training", {}).get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("runtime", {}).get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=verifier_cfg.get("training", {}).get("batch_size", 64),
        shuffle=False,
        num_workers=cfg.get("runtime", {}).get("num_workers", 4),
    )

    best_acc = 0.0
    patience = verifier_cfg.get("training", {}).get("early_stopping_patience", 15)
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if len(images) == 0:
                    continue
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f Acc: %.3f | Val Loss: %.4f Acc: %.3f | Time: %.1fs",
            epoch + 1, epochs, train_loss / max(len(train_loader), 1), train_acc,
            val_loss / max(len(val_loader), 1), val_acc, elapsed,
        )

        # Checkpoint
        if val_acc > best_acc or epoch % 5 == 0:
            best_acc = max(best_acc, val_acc)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, save_dir / "mobilenetv3_verifier_best.pt")
            torch.save(checkpoint, save_dir / "mobilenetv3_verifier.pt")
            torch.save(checkpoint, save_dir / f"mobilenetv3_verifier_epoch_{epoch}.pt")
            logger.info("Saved checkpoint (val_acc=%.3f)", val_acc)

        patience_counter += 1
        if patience_counter >= patience and epoch > 5:
            logger.info("Early stopping at epoch %d", epoch)
            break

    torch.save(model.state_dict(), save_dir / "mobilenetv3_verifier_final.pt")
    logger.info("Training complete. Best val_acc: %.3f", best_acc)


def main():
    train()


if __name__ == "__main__":
    main()
