# NEMO — Validated MVP Architecture

## What model is used?

The NEMO MVP uses a two-stage detection pipeline:

1. **Stage 1 — YOLO11n-1C:** Object localization and class prediction on native 1-channel sonar images
2. **Stage 2 — MobileNetV3-Small:** Binary candidate verification (target vs. background)

## What input representation is used?

Native 1-channel sonar acoustic-intensity images, resized/letterboxed to 640×640 pixels. No normalization or contrast enhancement is applied in the validated default configuration.

## Why was this architecture selected?

The architecture was selected based on controlled ablation experiments (T1–T4):

- **T1:** Native 1-channel input outperformed 3-channel RGB and preprocessed variants (mAP50: 0.5939 vs 0.4293 vs 0.4931)
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
