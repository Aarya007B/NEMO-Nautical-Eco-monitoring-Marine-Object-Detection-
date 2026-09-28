#!/usr/bin/env bash
# ==============================================================================
# train_all.sh — NEMO Local Two-Stage Model Training Script
#
# Target Hardware: Apple Silicon (M4 / MPS) or CUDA
# Stages:
#   1. Train Stage-1 YOLO11n Detector (SeabedObjects)
#   2. Train Stage-2 MobileNetV3-Small Verifier (Marine_PULSE 2)
#   3. Deploy checkpoints to models/*/weights/ and run validation
# ==============================================================================

set -e

# Change to repository root
cd "$(dirname "$0")"

echo "============================================================"
echo " NEMO — Two-Stage Model Training Pipeline"
echo "============================================================"

# 1. Detect Device
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
    echo "[Device] Apple Silicon MPS detected (Metal acceleration enabled)"
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="0"
    echo "[Device] NVIDIA CUDA detected (GPU 0 enabled)"
else
    echo "[Device] No accelerator detected; using CPU"
fi

# 2. Stage-1 Training: YOLO11n Detector
echo ""
echo "------------------------------------------------------------"
echo "[Stage 1/2] Training YOLO11n Detector on SeabedObjects"
echo "------------------------------------------------------------"

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

mkdir -p models/yolo11/weights/
cp runs/train/nemo_yolo11n/weights/best.pt models/yolo11/weights/yolo11n_1c.pt
echo "✓ Stage 1 Complete: Saved to models/yolo11/weights/yolo11n_1c.pt"

# 3. Stage-2 Training: MobileNetV3-Small Verifier
echo ""
echo "------------------------------------------------------------"
echo "[Stage 2/2] Training MobileNetV3-Small Verifier on Marine_PULSE 2"
echo "------------------------------------------------------------"

python3 -m models.mobilenetv3.train \
    --config configs/config.yaml \
    --data "data/raw/opensonardatasets/Marine_PULSE 2" \
    --epochs 50 \
    --output models/mobilenetv3/weights

if [ -f "models/mobilenetv3/weights/mobilenetv3_verifier_best.pt" ]; then
    cp models/mobilenetv3/weights/mobilenetv3_verifier_best.pt models/mobilenetv3/weights/mobilenetv3_verifier.pt
fi
echo "✓ Stage 2 Complete: Saved to models/mobilenetv3/weights/mobilenetv3_verifier.pt"

# 4. Pipeline Verification
echo ""
echo "------------------------------------------------------------"
echo "[Verification] Running System Smoke Test & Test Suite"
echo "------------------------------------------------------------"

python3 tools/smoke_test.py
python3 -m pytest tests/ -v

echo ""
echo "============================================================"
echo " All models successfully trained, deployed, and verified!"
echo " Start the API server with: python main.py --mode api"
echo "============================================================"
