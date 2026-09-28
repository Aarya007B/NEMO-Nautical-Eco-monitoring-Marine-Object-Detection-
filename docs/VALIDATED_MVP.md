# NEMO — Validated MVP Architecture

## What model is used?

The NEMO MVP uses a two-stage detection pipeline:

1. **Stage 1 — YOLO11n-1C:** Object localization and candidate class prediction on native 1-channel sonar images.
2. **Stage 2 — MobileNetV3-Small:** Binary candidate verification (target vs. background).

### YOLO11n-1C Architecture Definition: Native 1-Channel Adaptation

The validated NEMO detector is **YOLO11n with a native 1-channel input layer**. The model itself was adapted to accept a single-channel tensor `[B, 1, 640, 640]`.

The first convolution was explicitly adapted from:
$$\text{Conv2d}(3, 16, \text{kernel\_size}=3, \text{stride}=2, \text{padding}=1, \text{bias}=\text{False})$$
to:
$$\text{Conv2d}(1, 16, \text{kernel\_size}=3, \text{stride}=2, \text{padding}=1, \text{bias}=\text{False})$$

#### Weight Initialization Formula:
For the original pretrained first convolution with weights:
$$W \in \mathbb{R}^{C_{\text{out}} \times 3 \times k \times k}$$
the new native 1-channel weights were initialized by averaging across the input-channel dimension:
$$W_{\text{new}} = \text{mean}(W, \text{dim}=\text{channel}) \in \mathbb{R}^{C_{\text{out}} \times 1 \times k \times k}$$

**Pretrained Weights Claim:** The native 1-channel first-layer weights were initialized from the original pretrained 3-channel filters by averaging across the input-channel dimension, while the remaining compatible pretrained layers were retained. First-layer parameters are adapted, not unchanged.

#### Three Distinct Model Configurations:
- **CASE A — Original 3-channel YOLO:**
  - Input: `[B, 3, H, W]`
  - Model: `Conv2d(3, 16, ...)`
  - Standard optical RGB model.
- **CASE B — Grayscale replicated to 3 channels:**
  - Input: `[I, I, I]`
  - Model: `Conv2d(3, 16, ...)`
  - 3-channel model architecture receiving replicated data.
- **CASE C — NEMO YOLO11n-1C (Validated Architecture):**
  - Input: `[B, 1, H, W]`
  - Model: `Conv2d(1, 16, ...)`
  - Native 1-channel architecture (T1-B validated MVP).

NEMO's selected detector is strictly **CASE C**.

#### Data Flow:
```text
Raw SSS image
   ↓
Native single-channel image
   ↓
640 × 640 resizing / letterboxing
   ↓
Tensor [B, 1, 640, 640]
   ↓
YOLO11n-1C (Conv2d(1, 16, 3, 3))
   ↓
Bounding boxes + class + detector confidence
```

## What input representation is used?

Native 1-channel sonar acoustic-intensity images, resized/letterboxed to 640×640 pixels. No normalization or contrast enhancement is applied in the validated default configuration.

## Why was this architecture selected?

The architecture was selected based on controlled ablation experiments (T1–T4):

- **T1:** Native 1-channel input (CASE C, T1-B) outperformed 3-channel RGB and preprocessed variants (mAP50: 0.5939 vs 0.4293 vs 0.4931)
- **T2:** YOLO11n (nano) outperformed YOLO11s and YOLO11m on the available training data (mAP50: 0.5939 vs 0.5044 vs 0.3918)
- **T3:** Raw native input outperformed all preprocessing variants including normalization and CLAHE
- **T4:** Adding MobileNetV3-Small reduced false positives by ~15.79% while retaining recall

## What are the measured results?

### Detection Performance (KLSG Frozen Test Set)

| Configuration | Precision | Recall | F1 | FP | FN |
|---------------|-----------|--------|------|----|----|  
| YOLO11n-1C only | 0.7031 | 0.8654 | 0.7759 | 19 | 7 |
| YOLO11n-1C + MobileNetV3 | 0.7377 | 0.8654 | 0.7965 | 16 | 7 |

### Latency (NVIDIA Tesla T4)

| Pipeline | Latency | FPS |
|----------|---------|-----|
| YOLO11n-1C only | ~15.6 ms/image | ~64 FPS |
| YOLO11n-1C + MobileNetV3 | ~27.2 ms/image | ~37 FPS |

## What does MobileNetV3 add?

MobileNetV3-Small acts as a binary candidate verifier. After YOLO11n-1C produces candidate bounding boxes, each candidate crop is classified as "target" or "background" by MobileNetV3. This reduced false positives from 19 to 16 in the T4 evaluation while retaining all true positive detections.

The verifier is NOT the primary detector. YOLO performs localization; MobileNetV3 provides filtering.

## What is not yet validated?

- RCDI-YOLO comparative training study (no controlled benchmark completed)
- Acoustic evidence fusion model selection (rule-based scoring is implemented but not experimentally validated as a model-selection mechanism)
- Temporal consistency across sequential pings
- TensorRT FP16 optimization
- Adaptive hard-negative mining (infrastructure exists but repeated mining cycles not completed)
- Detection of debris categories beyond those in the KLSG training data
- Confidence calibration (confidence scores are not statistically calibrated)

## What remains future work?

See `docs/FUTURE_WORK.md` for the complete research roadmap.

## Central Conclusion

> Based on the completed controlled experiments, the NEMO submission MVP uses native 1-channel sonar input with YOLO11n-1C followed by MobileNetV3-Small verification. This is the architecture selected for this project's current validated MVP — not a claim of scientific universal optimality.
