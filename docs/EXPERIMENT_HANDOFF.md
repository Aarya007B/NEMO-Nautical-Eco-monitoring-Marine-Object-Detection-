# NEMO — Experiment Handoff Record

> **Authoritative record of completed experiments and architecture decisions.**
> These experiments were performed in a separate Colab runtime.
> The results below are the definitive reference — do not re-derive or assume alternative values.

---

## 1. Input Representation (T1)

**Selected:** Native grayscale 1-channel sonar input. No Normalize + CLAHE in the default production pipeline.

| Variant | mAP50 | mAP50-95 | Precision | Recall | Latency |
|---------|-------|----------|-----------|--------|---------|
| YOLO11n 3C baseline | 0.4293 | 0.2355 | 0.6892 | 0.3762 | ≈14.0 ms |
| YOLO11n native 1C | 0.5939 | 0.3062 | 0.6787 | 0.5259 | ≈14.54 ms (~68.8 FPS) |
| YOLO11n 1C + Normalize + CLAHE | 0.4931 | 0.3194 | 0.3771 | 0.5660 | ≈18.26 ms |

**Default preprocessing = native 1C.**

---

## 2. Detector Scale Comparison (T2)

| Model | Precision | Recall | mAP50 | mAP50-95 | Latency | FPS | Params |
|-------|-----------|--------|-------|----------|---------|-----|--------|
| YOLO11n-1C | 0.6787 | 0.5259 | 0.5939 | 0.3062 | ≈14.54 ms | ~68.8 | — |
| YOLO11s-1C | 0.5293 | 0.4673 | 0.5044 | 0.2836 | ≈19.82 ms | ~50.44 | ≈9.43M |
| YOLO11m-1C | 0.5984 | 0.3361 | 0.3918 | 0.2066 | ≈33.21 ms | ~30.11 | ≈20.05M |

**Selected detector = YOLO11n-1C for MVP.**

---

## 3. Preprocessing Ablation (T3)

| Preprocessing | Precision | Recall | mAP50 | mAP50-95 | Latency | FPS |
|--------------|-----------|--------|-------|----------|---------|-----|
| Raw native 1C | 0.6787 | 0.5259 | 0.5939 | 0.3062 | ≈14.54 ms | ~68.8 |
| Normalize-only | 0.4609 | 0.4610 | 0.4997 | 0.2995 | ≈25.97 ms | — |
| Normalize + CLAHE | 0.3771 | 0.5660 | 0.4931 | 0.3194 | ≈18.26 ms | — |
| Denoise + Normalize + CLAHE | 0.6715 | 0.4650 | 0.4892 | 0.3059 | ≈13.51 ms | ~74.03 |

Production default remains native 1C because the T1 controlled comparison established it as the selected representation.

---

## 4. MobileNetV3-Small Verifier

**Role:** Second-stage binary verifier. Does NOT replace YOLO classification.

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

---

## 5. T4 Pipeline Results

Evaluation was performed on the fixed KLSG test set, restricted to the relevant aircraft + shipwreck target subset.

> **These are NOT full four-class detector mAP figures. They are a relevant-target subset evaluation.**

### YOLO11n-1C Only

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

FP reduction on this fixed evaluation = 19 → 16. Recall was retained.

### T4 End-to-End Benchmark (NVIDIA Tesla T4)

| Pipeline | Latency (ms/image) | FPS |
|----------|-------------------|-----|
| YOLO11n-1C only | ≈15.624 | ~64.00 |
| YOLO11n-1C + MobileNetV3 | ≈27.185 | ~36.78 |

---

## 6. Final MVP Architecture

```
Native 1C sonar
  → YOLO11n-1C detector
  → candidate detections
  → MobileNetV3-Small verifier
  → confidence / priority scoring
  → GPS + mission metadata
  → MongoDB Atlas
  → NEMO dashboard
  → JSON / CSV reporting
```

---

## 7. RCDI-YOLO Status

RCDI-YOLO code exists in the repository as experimental/research infrastructure.

- **NOT the default detector**
- **NOT experimentally validated for NEMO**
- External paper results must NOT be presented as NEMO results
- The final MVP detector is **YOLO11n-1C**

---

## 8. Research Features (Not Required for MVP)

The following are preserved as infrastructure/research code but are NOT required for MVP operation:

- Acoustic evidence fusion (rule-based scoring IS implemented for operational reporting)
- Temporal consistency
- Knowledge distillation
- Adaptive preprocessing
- TensorRT-specific optimization
- Transformer-based detector variants
- Advanced segmentation

---

## 9. Database

- **Database:** MongoDB Atlas
- **Database name:** `nemo`
- **Collections:** `missions`, `detections`, `reports`
- **Client:** PyMongo from the backend
- **Credentials:** Environment variables (never exposed to frontend)

---

## 10. Branding

The project name is **NEMO**. No other product names.

---

## 11. Implementation Rules

1. Do not redesign the research architecture based on assumptions about missing Colab experiments.
2. Treat the values and decisions in this document as the completed experiment record.
3. Do not claim results that were not measured.
4. Do not present RCDI paper benchmarks as NEMO benchmarks.
5. Model checkpoint paths must be configurable, not hard-coded.
6. No random/untrained model fallback in normal inference — only explicit mock/demo mode.
