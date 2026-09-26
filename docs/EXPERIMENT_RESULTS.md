# NEMO — Experiment Results

This document records the measured experimental results from the NEMO architecture selection process. All experiments were performed on the KLSG / SeabedObjects dataset using mission-grouped splits.

> **These are measured results from controlled experiments. Do not alter these numbers.**

---

## T1: Input Representation Comparison

**Objective:** Compare 3-channel RGB, native 1-channel, and preprocessed 1-channel input representations for YOLO11n detection.

**Dataset:** KLSG / SeabedObjects validation set

| Variant | mAP50 | mAP50-95 | Precision | Recall | Latency (ms) |
|---------|-------|----------|-----------|--------|-------------|
| T1-A: YOLO11n 3-channel | 0.4293 | 0.2355 | 0.6892 | 0.3762 | ~14.0 |
| T1-B: YOLO11n native 1-channel | 0.5939 | 0.3062 | 0.6787 | 0.5259 | ~14.54 |
| T1-C: 1-channel + normalization + CLAHE | 0.4931 | 0.3194 | 0.3771 | 0.5660 | ~18.26 |

**Conclusion:** Native 1-channel input (T1-B) achieved the highest mAP50 (0.5939) with the best precision-recall balance. The preprocessed variant (T1-C) increased recall but at the cost of significantly reduced precision (0.3771). Native 1-channel was selected as the MVP input representation.

---

## T2: Model Scale Comparison

**Objective:** Compare YOLO11 model scales (nano, small, medium) with native 1-channel input.

**Dataset:** KLSG / SeabedObjects validation set

| Model | mAP50 | mAP50-95 | Precision | Recall | Latency (ms) | FPS |
|-------|-------|----------|-----------|--------|-------------|-----|
| YOLO11n-1C | 0.5939 | 0.3062 | 0.6787 | 0.5259 | ~14.54 | ~68.8 |
| YOLO11s-1C | 0.5044 | 0.2836 | 0.5293 | 0.4673 | ~19.82 | ~50.4 |
| YOLO11m-1C | 0.3918 | 0.2066 | 0.5984 | 0.3361 | ~33.21 | ~30.1 |

**Conclusion:** YOLO11n-1C (nano) achieved the best performance across all metrics and the highest throughput. Larger models did not improve detection quality on this dataset, likely due to limited training data. YOLO11n-1C was selected as the MVP detector.

---

## T3: Preprocessing Ablation

**Objective:** Evaluate the impact of preprocessing variants on YOLO11n-1C detection performance.

**Dataset:** KLSG / SeabedObjects validation set

| Preprocessing | mAP50 | mAP50-95 | Precision | Recall | Latency (ms) |
|--------------|-------|----------|-----------|--------|-------------|
| Raw native 1C | 0.5939 | 0.3062 | 0.6787 | 0.5259 | ~14.54 |
| Normalize-only | 0.4997 | 0.2995 | 0.4609 | 0.4610 | ~25.97 |
| Normalize + CLAHE | 0.4931 | 0.3194 | 0.3771 | 0.5660 | ~18.26 |
| Denoise + normalize + CLAHE | 0.4892 | 0.3059 | 0.6715 | 0.4650 | ~13.51 |

**Conclusion:** Raw native 1-channel input achieved the highest mAP50 and the best precision-recall trade-off. Normalization and contrast enhancement did not improve detection quality for YOLO11n-1C on this dataset. The preprocessing framework supports multiple enhancement strategies, but based on these results, the validated MVP uses native input with standard resizing/letterboxing.

---

## T4: Two-Stage Verification (YOLO11n-1C + MobileNetV3-Small)

**Objective:** Evaluate the impact of adding MobileNetV3-Small as a second-stage verifier to the YOLO11n-1C detector.

**Dataset:** KLSG frozen test set, restricted to the relevant aircraft + shipwreck target subset.

> **These are NOT full four-class detector mAP figures. They are a relevant-target subset evaluation.**

### Detector-only (YOLO11n-1C)

| Metric | Value |
|--------|-------|
| Precision | 0.7031 |
| Recall | 0.8654 |
| F1 | 0.7759 |
| TP | 45 |
| FP | 19 |
| FN | 7 |

### YOLO11n-1C + MobileNetV3-Small

| Metric | Value |
|--------|-------|
| Precision | 0.7377 |
| Recall | 0.8654 |
| F1 | 0.7965 |
| TP | 45 |
| FP | 16 |
| FN | 7 |

### Observed Change

- False positives reduced from 19 to 16 (3 removed)
- Measured recall retained at 0.8654
- Approximately 15.79% FP reduction on this evaluation
- This result was obtained on the frozen KLSG test set
- This is an experiment result, not a universal guarantee

### T4 End-to-End Benchmark (NVIDIA Tesla T4)

| Pipeline | Latency (ms/image) | FPS |
|----------|-------------------|-----|
| YOLO11n-1C only | ~15.624 | ~64.00 |
| YOLO11n-1C + MobileNetV3-Small | ~27.185 | ~36.78 |

**Hardware:** NVIDIA Tesla T4

**Conclusion:** The MobileNetV3-Small verifier reduced false positives by ~15.79% while retaining measured recall. The two-stage pipeline adds approximately 11.5 ms of latency per image, resulting in ~36.78 FPS end-to-end on a Tesla T4. This tradeoff is acceptable for the operational use case.

---

## Verifier Training Details

**Role:** MobileNetV3-Small is a second-stage binary verifier. It does NOT replace YOLO classification.

### Training Labels

| Category | Label |
|----------|-------|
| Positive | aircraft, shipwreck |
| Negative | background |
| Excluded | fish, other |

### Verifier Dataset

| Split | Positives | Background Negatives |
|-------|-----------|---------------------|
| Train | 409 | 810 |
| Validation | 100 | 232 |

### Best Verifier Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9940 |
| Precision | 1.0000 |
| Recall | 0.9800 |
| F1 | 0.9899 |

MobileNetV3-Small acts as a binary candidate verifier that distinguishes relevant target crops from background crops. It is NOT the primary object detector — YOLO11n-1C performs object localization and class prediction.

---

## Notes

1. All experiments used mission-grouped train/val/test splits to prevent spatial correlation data leakage.
2. Experiment numbers are reported as measured. No cherry-picking or post-hoc selection.
3. The RCDI-YOLO architecture was considered as a research candidate but was not included in the validated MVP because a controlled apples-to-apples trained benchmark comparing RCDI-YOLO to YOLO11n on the same dataset was not completed.
4. T4 results are a relevant-target subset evaluation, not full four-class detector mAP.
