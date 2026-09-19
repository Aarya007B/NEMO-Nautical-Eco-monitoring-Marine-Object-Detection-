"""
tools/create_split.py — Create mission-grouped train/val/test splits.

Splits data by mission (not by individual image) to prevent data leakage
from correlated consecutive sonar frames.

Usage:
    python tools/create_split.py --config configs/config.yaml --data data/processed
"""
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def discover_missions(data_dir: str) -> Dict[str, List[str]]:
    """Group image paths by mission (parent directory name)."""
    missions: Dict[str, List[str]] = {}
    root = Path(data_dir)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() in extensions:
            mission_id = img_path.parent.name
            missions.setdefault(mission_id, []).append(str(img_path))

    return missions


def split_missions(
    missions: Dict[str, List[str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split missions into train/val/test groups."""
    mission_ids = sorted(missions.keys())
    random.seed(seed)
    random.shuffle(mission_ids)

    n = len(mission_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_missions = mission_ids[:n_train]
    val_missions = mission_ids[n_train:n_train + n_val]
    test_missions = mission_ids[n_train + n_val:]

    splits = {"train": [], "val": [], "test": []}
    for m in train_missions:
        splits["train"].extend(missions[m])
    for m in val_missions:
        splits["val"].extend(missions[m])
    for m in test_missions:
        splits["test"].extend(missions[m])

    return splits


def discover_roboflow_splits(data_dir: str) -> Optional[Dict[str, List[str]]]:
    """Check if directory has Roboflow train / valid (or val) / test folders."""
    root = Path(data_dir)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    train_dir = root / "train"
    val_dir = root / "valid" if (root / "valid").is_dir() else root / "val"
    test_dir = root / "test"

    if not train_dir.is_dir():
        return None

    def gather_images(d: Path) -> List[str]:
        if not d.is_dir():
            return []
        img_dir = d / "images" if (d / "images").is_dir() else d
        return [str(p) for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in extensions]

    splits = {
        "train": gather_images(train_dir),
        "val": gather_images(val_dir) if val_dir.is_dir() else [],
        "test": gather_images(test_dir) if test_dir.is_dir() else [],
    }

    if not splits["train"]:
        return None

    if not splits["val"] and len(splits["train"]) > 5:
        n_val = max(1, int(len(splits["train"]) * 0.15))
        splits["val"] = splits["train"][-n_val:]
        splits["train"] = splits["train"][:-n_val]

    return splits


def main():
    parser = argparse.ArgumentParser(description="Create Mission-Grouped Splits")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/processed")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {}).get("split", {})
    train_ratio = data_cfg.get("train", 0.70)
    val_ratio = data_cfg.get("val", 0.15)
    seed = data_cfg.get("seed", 42)
    splits_dir = Path(config.get("data", {}).get("splits_dir", "data/splits"))

    # Check for Roboflow structure first
    splits = discover_roboflow_splits(args.data)
    if splits:
        logger.info("Detected Roboflow dataset format in %s", args.data)
    else:
        missions = discover_missions(args.data)
        if not missions:
            logger.warning("No images found in %s", args.data)
            return

        logger.info("Found %d missions, %d total images",
                    len(missions), sum(len(v) for v in missions.values()))

        splits = split_missions(missions, train_ratio, val_ratio, seed)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, paths in splits.items():
        filepath = splits_dir / f"{split_name}.txt"
        filepath.write_text("\n".join(paths) + "\n" if paths else "")
        logger.info("  %s: %d images -> %s", split_name, len(paths), filepath)


if __name__ == "__main__":
    main()
