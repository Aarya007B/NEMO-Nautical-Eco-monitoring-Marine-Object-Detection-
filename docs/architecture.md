# SonarGuard / NEMO — Architecture

## System Overview

```
Side-Scan Sonar -> Preprocessing -> RCDI-YOLO 1C -> Crop -> MobileNetV3 -> Fusion -> NEMO
```

## Design Rationale

### Why RCDI-YOLO?

| Sonar Challenge | RCDI Component |
|-----------------|----------------|
| Low contrast targets | LANConvNeXtv2 (multi-scale dilated conv) |
| Variable target size | LANConvNeXtv2 (dilation 1, 2, 4) |
| Noisy backgrounds | LANConvNeXtv2 (channel attention) |
| Blurred edges | DySample (adaptive upsampling) |
| Noisy features | ImplicitHead (ImplicitA/M) |

### Why MobileNetV3?

Stage-1 prioritizes recall. Stage-2 reduces false positives. MobileNetV3-Small is lightweight, independently trainable, and replaceable.

### Why Evidence Fusion?

A single confidence score is insufficient. Fusion combines detector confidence, verifier probability, and acoustic features.

### Why Separate Artificialness and Priority?

These are different concepts. High artificialness + well-mapped area = low priority. Medium artificialness + sensitive area = high priority.

### Why Hard Negatives?

Natural structures (rocks, ridges, coral, shadows) are the primary false positive source.

### Why Mission-Grouped Splits?

Consecutive sonar frames are correlated. Image-level random splits leak data.

## Module Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| Preprocessing | `preprocessing/` | Raw image -> [1,640,640] tensor |
| Detector | `models/rcdi_yolo/` | Candidate detection |
| Verifier | `models/mobilenetv3/` | Candidate verification |
| Data Adapters | `data_adapters/` | Dataset normalization |
| Inference Pipeline | `inference/` | End-to-end orchestration |
| Dashboard | `dashboard/` | NEMO Streamlit UI |
