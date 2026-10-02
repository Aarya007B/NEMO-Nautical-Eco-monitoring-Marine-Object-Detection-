"""
tools/convert_ai4shipwrecks.py — Tile AI4Shipwrecks SSS strips + masks to YOLO detection format.

Input (downloaded from https://umfieldrobotics.github.io/ai4shipwrecks/):
    <src>/{train,test}/images/*.png
    <src>/{train,test}/labels/*.png  (0/1 masks)
    <src>/extras/terrain/{images,labels}/*.png (all-zero masks)

Output (NEMO YOLO format, SeabedObjects taxonomy: 3=shipwreck):
    data/raw/ai4shipwrecks/{train,valid}/{images,labels}/
    data/raw/ai4shipwrecks/data.yaml

Strategy:
    - Tile tall strips (Hx1728) into TILExTILE patches, stride = TILE*(1-OVERLAP)
    - Per patch: connected components on mask -> one YOLO box per component
    - Empty patches -> empty .txt (background negatives); subsampled for train strips,
      all kept for terrain extras (hard negatives)
    - AI4 train -> YOLO train, AI4 test -> YOLO valid, terrain -> YOLO train

Usage:
    python tools/convert_ai4shipwrecks.py [--tile 1024] [--overlap 0.2] [--src ~/Downloads/AI4Shipwrecks]
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SRC = Path.home() / "Downloads" / "AI4Shipwrecks"
REPO = Path(__file__).resolve().parent.parent
DST = REPO / "data" / "raw" / "ai4shipwrecks"

SHIPWRECK_CLS = 3
MIN_BOX_FRAC = 0.0005  # min component area as fraction of tile (filters speckle)


def mask_to_boxes(mask: np.ndarray):
    """Connected components -> list of (cx,cy,w,h) normalized to mask size."""
    h, w = mask.shape[:2]
    bin_m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_BOX_FRAC * h * w:
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        boxes.append((cx, cy, bw / w, bh / h))
    return boxes


def tile_strip(img, mask, tile, stride):
    """Yield (x0, y0, img_patch, mask_patch)."""
    H, W = img.shape[:2]
    for y0 in list(range(0, max(H - tile + 1, 1), stride)) + ([H - tile] if H > tile else [0]):
        for x0 in list(range(0, max(W - tile + 1, 1), stride)) + ([W - tile] if W > tile else [0]):
            y0c = max(min(y0, H - tile), 0) if H >= tile else 0
            x0c = max(min(x0, W - tile), 0) if W >= tile else 0
            ip = img[y0c:y0c + tile, x0c:x0c + tile]
            mp = mask[y0c:y0c + tile, x0c:x0c + tile] if mask is not None else None
            # pad if strip smaller than tile (shouldn't happen, but safe)
            if ip.shape[0] != tile or ip.shape[1] != tile:
                ip = cv2.copyMakeBorder(ip, 0, tile - ip.shape[0], 0, tile - ip.shape[1],
                                        cv2.BORDER_CONSTANT, value=0)
                if mp is not None:
                    mp = cv2.copyMakeBorder(mp, 0, tile - mp.shape[0], 0, tile - mp.shape[1],
                                            cv2.BORDER_CONSTANT, value=0)
            yield x0c, y0c, ip, mp


def process_split(src_images, src_labels, out_images, out_labels, tile, stride,
                  keep_empty_every, split_name):
    n_pos, n_neg, n_skip = 0, 0, 0
    img_paths = sorted(src_images.glob("*.png"))
    logger.info("%s: %d strips", split_name, len(img_paths))
    for img_p in img_paths:
        lab_p = src_labels / img_p.name
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(lab_p), cv2.IMREAD_UNCHANGED) if lab_p.exists() else None
        if img is None:
            logger.warning("unreadable %s", img_p)
            continue
        if mask is not None and mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        stem = img_p.stem
        for x0, y0, ip, mp in tile_strip(img, mask, tile, stride):
            boxes = mask_to_boxes(mp) if mp is not None else []
            is_empty = len(boxes) == 0
            if is_empty:
                n_skip += 1
                if (n_skip % keep_empty_every) != 0:
                    continue
            out_img = out_images / f"{stem}_x{x0}_y{y0}.jpg"
            out_txt = out_labels / f"{stem}_x{x0}_y{y0}.txt"
            cv2.imwrite(str(out_img), ip, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(out_txt, "w") as f:
                for (cx, cy, w, h) in boxes:
                    f.write(f"{SHIPWRECK_CLS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            if is_empty:
                n_neg += 1
            else:
                n_pos += 1
    logger.info("%s: pos_tiles=%d neg_tiles=%d (empty skipped %d:1 sampled)",
                split_name, n_pos, n_neg, keep_empty_every)
    return n_pos, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="Downloaded AI4Shipwrecks root (with train/, test/, extras/).")
    args = ap.parse_args()
    src = Path(args.src).expanduser()
    if not (src / "train" / "images").exists():
        raise SystemExit(
            f"AI4Shipwrecks source not found at {src}. Download it from "
            "https://umfieldrobotics.github.io/ai4shipwrecks/ "
            "and pass --src <path>."
        )
    tile, stride = args.tile, int(args.tile * (1 - args.overlap))

    for d in [DST / "train" / "images", DST / "train" / "labels",
              DST / "valid" / "images", DST / "valid" / "labels"]:
        d.mkdir(parents=True, exist_ok=True)

    # AI4 train -> YOLO train (subsample empties 1:4, strips have plenty of water)
    process_split(src / "train" / "images", src / "train" / "labels",
                  DST / "train" / "images", DST / "train" / "labels",
                  tile, stride, keep_empty_every=4, split_name="ai4-train")
    # AI4 test -> YOLO valid (keep more negatives for honest val)
    process_split(src / "test" / "images", src / "test" / "labels",
                  DST / "valid" / "images", DST / "valid" / "labels",
                  tile, stride, keep_empty_every=2, split_name="ai4-test")
    # terrain extras (all background) -> YOLO train hard negatives, keep all
    if (src / "extras" / "terrain" / "images").exists():
        process_split(src / "extras" / "terrain" / "images",
                      src / "extras" / "terrain" / "labels",
                      DST / "train" / "images", DST / "train" / "labels",
                      tile, stride, keep_empty_every=1, split_name="terrain-neg")

    yaml_text = (
        "train: ../train/images\nvalid: ../valid/images\n\nnc: 4\n"
        "names: ['aircraft', 'fish', 'other', 'shipwreck']\n"
    )
    (DST / "data.yaml").write_text(yaml_text)
    logger.info("wrote %s", DST / "data.yaml")
    logger.info("DONE. All AI4 boxes are class 3=shipwreck (SeabedObjects taxonomy).")


if __name__ == "__main__":
    main()
