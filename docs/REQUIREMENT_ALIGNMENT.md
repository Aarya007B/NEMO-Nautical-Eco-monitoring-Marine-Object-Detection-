# NEMO — Organizer Requirement Alignment

This document maps the NEMO MVP capabilities to the organizer's competition requirements.

---

## 1. Object Detection / Semantic Segmentation

**Requirement:** Detect objects of interest in side-scan sonar imagery.

**NEMO Implementation:** YOLO11n-1C bounding-box detection on native 1-channel sonar images.

**Status:** ✅ Demonstrated — Bounding-box object detection with measured precision 0.7377 and recall 0.8654 on the KLSG test set.

**Note:** The MVP uses object detection (bounding boxes), not semantic segmentation (pixel masks). Semantic segmentation is listed as future work.

---

## 2. Confidence Scoring / Noise Filtering

**Requirement:** Provide confidence scores and filter noise/false positives.

**NEMO Implementation:**
- **YOLO detector confidence:** Per-detection sigmoid confidence from the detection head
- **MobileNetV3 verification:** Binary second-stage filtering (target vs. background)
- **Operational scoring:** Rule-based artificialness score and priority score for operator workflow

**Status:** ✅ Demonstrated — Two-stage pipeline reduced false positives by ~15.79% in T4 evaluation.

**Note:** Confidence scores are not statistically calibrated. The artificialness and priority scores are rule-based operational aids, not learned probabilistic outputs.

---

## 3. Anomalous Reporting / Geotagging

**Requirement:** Generate reports with geolocation for detected anomalies.

**NEMO Implementation:**
- GPS metadata alignment via navigation CSV logs
- Latitude/longitude association per detection
- JSON and CSV export with full detection metadata
- Per-detection records including bounding box, confidence, classification, GPS coordinates, and timestamp

**Status:** ✅ Implemented — Metadata alignment, GPS parsing, and structured reporting are functional.

---

## 4. UI Dashboard

**Requirement:** Provide a user interface for viewing and managing detections.

**NEMO Implementation:**
- FastAPI REST backend at `localhost:8000`
- NEMO dashboard (vanilla HTML/CSS/JS frontend)
- Mission processing, detection inspection, crop viewing
- JSON/CSV export from dashboard
- System health monitoring

**Dashboard workflow:**
1. Upload / select sonar mission
2. View sonar image
3. See detected targets with bounding boxes
4. See confidence scores and verification status
5. See GPS location where metadata exists
6. Inspect candidate crops
7. Export report (JSON/CSV)

**Status:** ✅ Implemented — Dashboard is functional with full API integration.

---

## What is Currently Demonstrated vs. Planned

| Capability | Status |
|-----------|--------|
| Bounding-box object detection | ✅ Demonstrated |
| Two-stage false positive filtering | ✅ Demonstrated |
| Confidence scoring | ✅ Implemented |
| GPS geotagging | ✅ Implemented |
| JSON/CSV reporting | ✅ Implemented |
| Dashboard UI | ✅ Implemented |
| REST API | ✅ Implemented |
| Semantic segmentation | 🔮 Planned |
| Real-time streaming | 🔮 Planned |
| Multi-class detection | 🔮 Planned (requires additional training data) |
| Temporal consistency | 🔮 Planned |
| TensorRT deployment | 🔮 Planned |

---

## Dataset Scope

NEMO currently demonstrates detection of target/anomaly classes represented in the validated sonar training data (KLSG / SeabedObjects). The architecture is extensible to additional marine debris classes as annotated datasets become available.

Other datasets in `data/raw/` are included as future expansion sources and were not used in the validated MVP experiments.
