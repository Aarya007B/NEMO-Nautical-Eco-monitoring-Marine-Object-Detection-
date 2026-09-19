# RCDI-YOLO Paper Reference

## Paper Traceability Document

---

## SOURCE: Published Paper

**Title:** RCDI-YOLO: a target-detection method for complex environment side-scan sonar images based on improved YOLOv8

**Authors:** Zhang, J. and Gao, B.

**Journal:** Frontiers in Marine Science, Volume 12, Article 1679077, 2025

**DOI:** 10.3389/fmars.2025.1679077

---

## SOURCE: Published Architecture

RCDI-YOLO is a modification of **YOLOv8** (not YOLO11).

```
R = LANConvNeXtv2
D = Dysample
I = ImplicitHead
```

### LANConvNeXtv2 Placement

| Location | Position | Replaces |
|---|---|---|
| Backbone | 1, 2 | C2f #1, C2f #2 |
| Neck | 1, 2, 4 | C2f #1, C2f #2, C2f #4 |

### DySample

Replaces fixed bilinear/nearest upsampling in the FPN neck.

### ImplicitHead

Replaces the standard YOLOv8 detection head. Combines ImplicitA, ImplicitM, and DFL.

---

## SOURCE: Reported Results (CESSSD dataset, RTX 3090)

> **These are the paper's own experimental results.**
> **They are NOT SonarGuard results.**

| Metric | Paper Value |
|--------|------------|
| Parameters | 3.23M |
| GFLOPs | 9.3 |
| Precision | 95.3% |
| Recall | 88.8% |
| mAP@0.5 | 95.7% |
| mAP@0.5:0.95 | 60.8% |
| FPS (RTX 3090) | ~163 |

---

## PROJECT ADAPTATION: SonarGuard Modifications

### Adaptation 1: 1-Channel Sonar Input

**SOURCE:** Published RCDI-YOLO uses 3-channel RGB input.

**SONARGUARD:** First conv changed to `in_channels=1` for sonar acoustic-intensity.

The resulting model is referred to as **SonarGuard RCDI detector** or **RCDI-YOLO-based 1C sonar detector**.

### Adaptation 2: Single Unified Class

For the MVP, all target objects map to a single `target` class.

### Adaptation 3: Second-Stage Verifier

SonarGuard adds MobileNetV3-Small as a second-stage verifier. Not part of the published RCDI-YOLO.

### Adaptation 4: Evidence Fusion

Configurable evidence fusion layer. Not in the paper.

---

## File Locations

| Component | Location |
|-----------|----------|
| LANConvNeXtv2 | `models/rcdi_yolo/modules/lanconvnextv2.py` |
| DySample | `models/rcdi_yolo/modules/dysample.py` |
| ImplicitA/M | `models/rcdi_yolo/modules/implicit.py` |
| ImplicitHead | `models/rcdi_yolo/modules/implicit_head.py` |
| Backbone | `models/rcdi_yolo/backbone.py` |
| Neck | `models/rcdi_yolo/neck.py` |
| Full Model | `models/rcdi_yolo/model.py` |
| Reference Config | `configs/reference/rcdi_paper.yaml` |
| Active Config | `configs/rcdi_yolo_1c.yaml` |

---

## Citation

```bibtex
@article{zhang2025rcdiyolo,
  author  = {Zhang, Jinhao and Gao, Bo},
  title   = {RCDI-YOLO: a target-detection method for complex environment
             side-scan sonar images based on improved YOLOv8},
  journal = {Frontiers in Marine Science},
  volume  = {12},
  pages   = {1679077},
  year    = {2025},
  doi     = {10.3389/fmars.2025.1679077}
}
```
