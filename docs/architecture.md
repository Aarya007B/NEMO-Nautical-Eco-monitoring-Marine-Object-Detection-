# NEMO — Architecture

## System Overview

```
Native 1C Sonar → Preprocessing → YOLO11n-1C → Crop → MobileNetV3 → Scoring → MongoDB → NEMO Dashboard
```

## Validated MVP Pipeline

| Stage | Component | Status |
|-------|-----------|--------|
| Input | Native 1-channel sonar | ✅ Validated (T1) |
| Preprocessing | Letterbox resize to [1, 640, 640] | ✅ Validated (T3) |
| Stage-1 Detector | YOLO11n-1C | ✅ Validated (T2, T4) |
| Crop Extraction | Padded bbox crop to 128×128 | ✅ Implemented |
| Stage-2 Verifier | MobileNetV3-Small | ✅ Validated (T4) |
| Scoring | Rule-based artificialness + priority | ✅ Implemented |
| Metadata | GPS/ping alignment | ✅ Implemented |
| Storage | MongoDB Atlas | ✅ Implemented |
| Reporting | JSON/CSV export | ✅ Implemented |
| Dashboard | FastAPI REST API + HTML frontend | ✅ Implemented |

## Subsystem Readiness

| Subsystem | Readiness |
|-----------|-----------|
| YOLO11n-1C detector | Validated — T1/T2/T4 experiments |
| MobileNetV3-Small verifier | Validated — T4 experiment |
| Native 1C preprocessing | Validated — T1/T3 experiments |
| Evidence fusion scoring | Implemented — rule-based, not ablated |
| GPS metadata alignment | Implemented — not field-tested |
| MongoDB storage | Implemented — not load-tested |
| RCDI-YOLO detector | Experimental — not validated for NEMO |
| Hard-negative mining | Infrastructure ready — cycles not completed |
| Temporal consistency | Not implemented |

## Design Rationale

### Why YOLO11n-1C?

YOLO11n (nano) achieved the highest mAP50 (0.5939) among tested scales (nano, small, medium) on the available sonar training data. Larger models did not improve accuracy, likely due to limited dataset size.

### Why MobileNetV3?

Stage-1 prioritizes recall. Stage-2 reduces false positives. MobileNetV3-Small is lightweight, independently trainable, and replaceable. T4 evaluation showed ~15.79% FP reduction with retained recall.

### Why Evidence Fusion?

A single confidence score is insufficient. Fusion combines detector confidence, verifier probability, and acoustic features into separate artificialness and priority scores.

### Why Separate Artificialness and Priority?

These are different concepts. High artificialness + well-mapped area = low priority. Medium artificialness + sensitive area = high priority.

### Why Mission-Grouped Splits?

Consecutive sonar frames are correlated. Image-level random splits leak data.

## Module Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| Preprocessing | `preprocessing/` | Raw image → [1, 640, 640] tensor |
| MVP Detector | `models/yolo11/` | YOLO11n-1C candidate detection |
| Verifier | `models/mobilenetv3/` | Candidate verification |
| Experimental Detector | `models/rcdi_yolo/` | RCDI-YOLO (not default) |
| Data Adapters | `data_adapters/` | Dataset normalization |
| Inference Pipeline | `inference/` | End-to-end orchestration |
| Database | `database/` | MongoDB Atlas storage |
| Dashboard | `dashboard/` | NEMO REST API + frontend |
