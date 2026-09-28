#!/usr/bin/env bash
# ==============================================================================
# train_verifier.sh — NEMO Stage-2 MobileNetV3-Small Verifier Training Script
#
# Hardware: Auto-detects Apple Silicon (MPS), CUDA, or CPU
# Dataset: Marine_PULSE 2 (and hard negatives)
# Target output: models/mobilenetv3/weights/mobilenetv3_verifier.pt
# ==============================================================================

set -e

# Change to repository root
cd "$(dirname "$0")"

echo "============================================================"
echo " NEMO — Stage 2: Training MobileNetV3-Small Verifier"
echo "============================================================"

# Train the verifier
python3 -m models.mobilenetv3.train \
    --config configs/config.yaml \
    --data "data/raw/opensonardatasets/Marine_PULSE 2" \
    --epochs 50 \
    --output models/mobilenetv3/weights

# Deploy best checkpoint
if [ -f "models/mobilenetv3/weights/mobilenetv3_verifier_best.pt" ]; then
    cp models/mobilenetv3/weights/mobilenetv3_verifier_best.pt models/mobilenetv3/weights/mobilenetv3_verifier.pt
    echo ""
    echo "============================================================"
    echo "✓ Stage-2 Verifier Training Successful!"
    echo "Best verifier weights deployed to: models/mobilenetv3/weights/mobilenetv3_verifier.pt"
    echo "============================================================"
else
    echo "Warning: best checkpoint not found; checking final weights."
fi
