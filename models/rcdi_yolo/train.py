"""
models/rcdi_yolo/train.py

RCDI-YOLO training entry point.

Usage:
    python -m models.rcdi_yolo.train --config configs/config.yaml --epochs 10
    python -m models.rcdi_yolo.train --config configs/config.yaml --weights path/to/weights.pt
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


def build_dataloader(config: dict, split: str = "train"):
    """Build a simple sonar dataset dataloader."""
    from torch.utils.data import DataLoader
    from models.rcdi_yolo.dataset import SonarDetectionDataset

    data_cfg = config.get("data", {})
    processed_dir = data_cfg.get("processed", "data/processed/images")
    splits_dir = data_cfg.get("splits_dir", "data/splits")

    split_file = Path(splits_dir) / f"{split}.txt"
    if not split_file.exists():
        logger.warning("Split file not found: %s", split_file)
        return None

    with open(split_file) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if not image_paths:
        logger.warning("No images in split: %s", split_file)
        return None

    # Find corresponding annotation files
    annotation_paths = []
    for img_path in image_paths:
        p = Path(img_path)
        ann_path = p.with_suffix(".txt")
        if not ann_path.exists():
            p_str = str(p)
            if "/images/" in p_str:
                cand = Path(p_str.replace("/images/", "/labels/")).with_suffix(".txt")
                if cand.exists():
                    ann_path = cand
        annotation_paths.append(str(ann_path) if ann_path.exists() else "")

    dataset = SonarDetectionDataset(
        image_paths=image_paths,
        annotation_paths=annotation_paths,
        image_size=config["input"].get("image_size", 640),
        num_classes=config["detector"].get("num_classes", 1),
        augment=(split == "train"),
    )

    batch_size = config["detector"].get("training", {}).get("batch_size", 16)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config.get("runtime", {}).get("num_workers", 4),
        collate_fn=lambda batch: batch,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if not batch:
            continue

        images = []
        targets = []
        for item in batch:
            if isinstance(item, dict):
                images.append(item.get("image"))
                targets.append(item.get("targets"))
            else:
                images.append(item[0])
                targets.append(item[1] if len(item) > 1 else None)

        if not images or images[0] is None:
            continue

        images_tensor = torch.cat(images, dim=0).to(device)

        predictions = model(images_tensor)

        try:
            gt_targets = torch.cat(targets, dim=0) if targets[0] is not None else None
            loss, loss_dict = criterion(predictions, gt_targets)
        except Exception as e:
            logger.debug("Loss computation issue: %s", e)
            loss = predictions[0].sum() * 0
            loss_dict = {"box": 0, "cls": 0, "dfl": 0}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(
                "Epoch %d, Batch %d, Loss: %.4f",
                epoch, batch_idx, loss.item(),
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_detections = 0
    total_targets = 0

    with torch.no_grad():
        for batch in dataloader:
            if not batch:
                continue
            images = []
            for item in batch:
                if isinstance(item, dict):
                    images.append(item.get("image"))
                else:
                    images.append(item[0])
            if not images or images[0] is None:
                continue
            images_tensor = torch.cat(images, dim=0).to(device)
            predictions = model(images_tensor)
            try:
                loss, loss_dict = criterion(predictions, None)
                total_loss += loss.item()
                num_batches += 1
            except Exception:
                pass

            for item in batch:
                if isinstance(item, dict):
                    targets = item.get("targets")
                    if targets is not None:
                        total_targets += len(targets)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, total_targets


def main():
    parser = argparse.ArgumentParser(description="RCDI-YOLO Trainer")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None, help="Path to raw or processed dataset root")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default="models/rcdi_yolo/weights")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("SonarGuard RCDI-YOLO Trainer")
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

    from models.rcdi_yolo.model import RCDIYOLOModel
    from models.rcdi_yolo.losses import RCDILoss
    from models.rcdi_yolo.parser import get_model_kwargs_from_file

    model_kwargs = get_model_kwargs_from_file(cfg["detector"]["config"])
    model = RCDIYOLOModel(**model_kwargs).to(device)
    logger.info("Model: %.2fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    criterion = RCDILoss(
        num_classes=model_kwargs["num_classes"],
        reg_max=model_kwargs["reg_max"],
        box_weight=cfg["detector"].get("training", {}).get("loss", {}).get("box", 7.5),
        cls_weight=cfg["detector"].get("training", {}).get("loss", {}).get("cls", 0.5),
        dfl_weight=cfg["detector"].get("training", {}).get("loss", {}).get("dfl", 1.5),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["detector"].get("training", {}).get("learning_rate", 0.001),
        weight_decay=cfg["detector"].get("training", {}).get("weight_decay", 0.0005),
    )

    epochs = args.epochs or cfg["detector"].get("training", {}).get("epochs", 100)
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

    # Also attempt to load detector weights if specified
    weights_path = cfg["detector"].get("weights", "")
    if weights_path and Path(weights_path).exists() and not args.resume:
        try:
            model.load_weights(weights_path)
            logger.info("Loaded pretrained weights: %s", weights_path)
        except Exception as e:
            logger.warning("Could not load weights: %s", e)

    # Auto-generate splits if dataset path is supplied or data/splits is missing
    splits_dir = Path(cfg.get("data", {}).get("splits_dir", "data/splits"))
    train_split_file = splits_dir / "train.txt"
    dataset_candidate = args.data or ("data/raw/seabedobjects" if Path("data/raw/seabedobjects").exists() else None)
    if (not train_split_file.exists() or args.data) and dataset_candidate and Path(dataset_candidate).exists():
        logger.info("Generating splits from %s...", dataset_candidate)
        from tools.create_split import discover_roboflow_splits, discover_missions, split_missions
        roboflow_splits = discover_roboflow_splits(dataset_candidate)
        if roboflow_splits:
            splits = roboflow_splits
            logger.info("Found Roboflow dataset format in %s", dataset_candidate)
        else:
            missions = discover_missions(dataset_candidate)
            splits = split_missions(missions)
        splits_dir.mkdir(parents=True, exist_ok=True)
        for s_name, s_paths in splits.items():
            (splits_dir / f"{s_name}.txt").write_text("\n".join(s_paths) + "\n" if s_paths else "")
        logger.info("Splits generated: train=%d, val=%d", len(splits.get("train", [])), len(splits.get("val", [])))

    # Build dataloader
    train_loader = build_dataloader(cfg, split="train")
    if train_loader is None:
        logger.warning(
            "No training data found. Run: python tools/create_split.py --config %s",
            args.config,
        )
        logger.warning("Falling back to synthetic training loop...")
        # Synthetic training smoke test
        for epoch in range(min(3, epochs)):
            fake_input = torch.randn(2, 1, 640, 640).to(device)
            fake_target = torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]])
            predictions = model(fake_input)
            loss, loss_dict = criterion(predictions, fake_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info("Epoch %d: synthetic loss=%.4f", epoch, loss.item())
        logger.info("Synthetic training complete.")
        return

    val_loader = build_dataloader(cfg, split="val")

    best_loss = float("inf")
    patience = cfg["detector"].get("training", {}).get("early_stopping_patience", 30)
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_targets = validate(model, val_loader, criterion, device) if val_loader else (0, 0)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Time: %.1fs",
            epoch + 1, epochs, train_loss, val_loss, elapsed,
        )

        # Checkpoint
        if epoch % 5 == 0 or val_loss < best_loss:
            best_loss = min(best_loss, val_loss)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": val_loss,
            }
            if val_loss < best_loss:
                torch.save(checkpoint, save_dir / "rcdi_yolo_1c_best.pt")
            torch.save(checkpoint, save_dir / f"rcdi_yolo_1c_epoch_{epoch}.pt")
            logger.info("Saved checkpoint at epoch %d", epoch)

        patience_counter += 1
        if patience_counter >= patience and epoch > 5:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Save final model
    torch.save(model.state_dict(), save_dir / "rcdi_yolo_1c_final.pt")
    logger.info("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
