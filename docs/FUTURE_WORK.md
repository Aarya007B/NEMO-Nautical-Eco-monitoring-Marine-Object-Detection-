# NEMO — Future Work

The following items represent research directions and engineering improvements that are preserved in the codebase but were not experimentally validated as part of the current MVP submission.

## Research Candidates

### RCDI-YOLO Comparative Training Study
The RCDI-YOLO architecture (Zhang & Gao, 2025) was implemented and is available at `models/rcdi_yolo/`. A controlled apples-to-apples comparison on the KLSG dataset — training RCDI-YOLO with identical data, splits, and hyperparameters as YOLO11n-1C — was not completed. This comparison would determine whether the RCDI modifications (LANConvNeXtv2, DySample, ImplicitHead) provide measurable improvement over standard YOLO11 on sonar data.

**Status:** Code complete, comparative benchmark pending  
**Location:** `models/rcdi_yolo/`, `configs/research_rcdi.yaml`

### Adaptive Hard-Negative Mining
Infrastructure for mining false-positive crops from challenging natural structures (rocks, ridges, shadows, coral formations) exists at `tools/mine_hard_negatives.py`. Repeated mining cycles and retraining were not completed for the MVP submission.

**Status:** Infrastructure ready, iterative refinement pending  
**Location:** `tools/mine_hard_negatives.py`, `data/hard_negatives/`

### Acoustic Evidence Feature Ablation
The evidence fusion layer combines detector confidence, verifier probability, and rule-based acoustic features. A systematic ablation of individual feature contributions (logistic regression, XGBoost, or learned fusion) was not performed.

**Status:** Rule-based scoring implemented, learned fusion pending  
**Location:** `scoring/`, `inference/evidence_fusion.py`

### Temporal Consistency
Sequential ping association and persistence analysis for tracking detections across consecutive sonar frames. This would enable temporal filtering of transient artifacts and improve confidence in persistent targets.

**Status:** Future extension  
**Location:** Not yet implemented

## Engineering Improvements

### TensorRT FP16 Optimization
Export YOLO11n-1C and MobileNetV3-Small to TensorRT FP16 for deployment on NVIDIA edge devices (Jetson). Expected to significantly improve throughput.

### Synthetic Augmentation Study
Evaluate domain-specific synthetic augmentation strategies (simulated sonar noise, shadow generation, target insertion) to expand the effective training set.

### Knowledge Distillation
Distill larger detection models (YOLO11m or RCDI-YOLO) into YOLO11n-1C to potentially improve nano-model accuracy without increasing inference cost.

### Transformer-Based Alternatives
Evaluate vision transformer detectors (RT-DETR, DINO) as alternatives to CNN-based detection for sonar imagery.

### Semantic Segmentation
Extend from bounding-box detection to pixel-level segmentation for more precise debris delineation. The current MVP uses object detection only.

### Expanded Debris Categories
Train multi-class detection covering additional marine debris types (ghost nets, UXO, pipelines, cables) as annotated datasets become available.

### Larger Multi-Dataset Training
Combine multiple sonar datasets (SeabedObjects, DFKI-UXO, OpenSonarDatasets, Kaggle SSS) for more robust cross-domain training.

### Live Sonar Stream Integration
Real-time streaming sonar processing. The mission source abstraction (`mission/source.py`) supports this extension. Validation of real-time streaming was not performed.
