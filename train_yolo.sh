#!/usr/bin/env bash
# ==============================================================================
# train_yolo.sh — NEMO Stage-1 YOLO11n-1C Detector Training Script
#
# Hardware: Auto-detects Apple Silicon (MPS), CUDA, or CPU
# Dataset: SeabedObjects (data/raw/seabedobjects/data.yaml)
# Target output: models/yolo11/weights/yolo11n_1c.pt
# ==============================================================================

set -e

# Change to repository root
cd "$(dirname "$0")"

echo "============================================================"
echo " NEMO — Stage 1: Training YOLO11n-1C Detector"
echo "============================================================"

# 1. Detect Accelerator
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
    echo "[Device] Apple Silicon Metal (MPS) detected & enabled"
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="0"
    echo "[Device] NVIDIA CUDA detected (GPU 0 enabled)"
else
    echo "[Device] No GPU accelerator detected; falling back to CPU"
fi

LAST_PT="runs/detect/runs/train/nemo_yolo11n/weights/last.pt"
if [ ! -f "$LAST_PT" ] && [ -f "runs/train/nemo_yolo11n/weights/last.pt" ]; then
    LAST_PT="runs/train/nemo_yolo11n/weights/last.pt"
fi

# Check if user requested --resume
RESUME=false
for arg in "$@"; do
    if [ "$arg" == "--resume" ] || [ "$arg" == "-r" ]; then
        RESUME=true
    fi
done

if [ "$RESUME" = true ] && [ -f "$LAST_PT" ]; then
    echo "[Resume] Resuming training from last checkpoint: $LAST_PT"
    yolo detect train resume model="$LAST_PT" amp=False
else
    echo "[Dataset] Using data/raw/seabedobjects/data.yaml (581 images)"
    echo "[Model]   yolo11n.pt (Transfer learning fine-tuning)"
    echo "[Params]  640x640 resolution, 100 epochs, batch size 16"
    echo ""
    # 2. Run Training (amp=False required for stable PyTorch MPS on Apple Silicon)
    yolo detect train \
        data="data/raw/seabedobjects/data.yaml" \
        model="yolo11n.pt" \
        imgsz=640 \
        epochs=100 \
        batch=16 \
        device="$DEVICE" \
        amp=False \
        project="runs/train" \
        name="nemo_yolo11n" \
        exist_ok=True
fi

# 3. Deploy Weights to Model Directory
mkdir -p models/yolo11/weights/
BEST_PT=""
if [ -f "runs/detect/runs/train/nemo_yolo11n/weights/best.pt" ]; then
    BEST_PT="runs/detect/runs/train/nemo_yolo11n/weights/best.pt"
elif [ -f "runs/train/nemo_yolo11n/weights/best.pt" ]; then
    BEST_PT="runs/train/nemo_yolo11n/weights/best.pt"
fi

if [ -n "$BEST_PT" ]; then
    cp "$BEST_PT" models/yolo11/weights/yolo11n_1c.pt
    echo ""
    echo "============================================================"
    echo "✓ Stage-1 Training Successful!"
    echo "Best model deployed to: models/yolo11/weights/yolo11n_1c.pt"
    echo "============================================================"
else
    echo "Error: best.pt was not found."
    exit 1
fi
