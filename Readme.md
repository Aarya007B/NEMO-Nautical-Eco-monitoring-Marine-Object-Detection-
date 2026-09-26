# NEMO 🌊
### AI-Powered Underwater Marine Debris & Anomaly Detection System

**NEMO** is a modular, two-stage AI pipeline that ingests side-scan sonar imagery, detects man-made marine debris and anomalies, and presents findings through the NEMO mission dashboard.

> **Do not treat every sonar detection as a confirmed object.** NEMO uses a two-stage pipeline to reduce false positives caused by natural seabed structures.

---

## Validated MVP Architecture

Selected based on controlled T1–T4 ablation experiments. See `docs/EXPERIMENT_RESULTS.md` for full results.

```
Native 1-Channel Sonar Input
        |
        v
   Preprocessing
   (grayscale → letterbox resize → [1, 640, 640])
        |
        v
   YOLO11n-1C                 ← Stage 1: Candidate Detection
   (validated MVP detector)
        |
        v
   Candidate Bounding Boxes
        |
        v
   Crop Extraction
        |
        v
   MobileNetV3-Small          ← Stage 2: Candidate Verification
   (target vs background)
        |
        v
   Confidence + Priority Scoring
        |
        v
   GPS / Ping Metadata Alignment
        |
        v
   MongoDB Atlas Storage
        |
       / \
      /   \
     v     v
  NEMO     JSON / CSV
Dashboard   Report
```

---

## Why This Architecture?

| Decision | Experiment | Result |
|----------|-----------|--------|
| Native 1C input (not RGB or preprocessed) | T1 | mAP50: 0.5939 vs 0.4293 (3C) vs 0.4931 (preprocessed) |
| YOLO11n (not YOLO11s or YOLO11m) | T2 | mAP50: 0.5939 vs 0.5044 vs 0.3918 |
| Raw input (no normalize/CLAHE) | T3 | Best precision-recall at raw native |
| +MobileNetV3 verifier | T4 | FP reduced 19→16, recall retained at 0.8654 |

---

## Two-Stage Pipeline

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | YOLO11n-1C | High recall — find all candidates |
| 2 | MobileNetV3-Small | Reduce false positives — target vs background |

Scores stored independently: `detector_confidence`, `verifier_artificial_prob`, `artificialness_score`, `priority_score`.

---

## Performance (Measured on NVIDIA Tesla T4)

| Configuration | Precision | Recall | F1 | FPS |
|---------------|-----------|--------|------|-----|
| YOLO11n-1C only | 0.7031 | 0.8654 | 0.7759 | ~64 |
| YOLO11n-1C + MobileNetV3 | 0.7377 | 0.8654 | 0.7965 | ~37 |

> These are measured results on the KLSG frozen test set (aircraft + shipwreck target subset). See `docs/EXPERIMENT_HANDOFF.md`.

---

## Getting Started

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables (for MongoDB Atlas)
```bash
export NEMO_MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
export NEMO_MONGO_DB="nemo"
```
> MongoDB is optional. Without it, results are stored locally via JSON/CSV export.

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Run Smoke Test (no GPU or data required)
```bash
python tools/smoke_test.py
```

---

## Dashboard (NEMO)

The dashboard uses a **FastAPI REST backend** with a vanilla HTML/JS frontend.

### Start the API server:
```bash
python main.py --mode api --config configs/config.yaml
```
Server runs at `http://localhost:8000`.

### Open the dashboard:
Open `dashboard/index.html` in a browser.

### API Endpoints:
```
GET  /api/health          — Health check (incl. MongoDB status)
GET  /api/status          — Model info and config
GET  /api/missions        — List available missions
POST /api/mission/process — Process a mission directory
GET  /api/mission/{id}    — Get mission results
GET  /api/mission/{id}/detections — Get detections
POST /api/detect          — Run detection on a single image
GET  /api/crop/{id}       — Get detection crop
GET  /api/export/json     — Export as JSON
GET  /api/export/csv      — Export as CSV
GET  /api/demo            — Get demo mission data
GET  /api/config          — Get current config
```

---

## Database

NEMO uses MongoDB Atlas for persistent storage.

| Collection | Contents |
|-----------|----------|
| `missions` | Processed mission records |
| `detections` | Per-frame detection results |
| `reports` | Generated report metadata |

Credentials come from environment variables (`NEMO_MONGO_URI`, `NEMO_MONGO_DB`). The browser/frontend **never** connects directly to MongoDB.

---

## Data Scope

NEMO currently demonstrates detection on the KLSG / SeabedObjects dataset (aircraft and shipwreck targets). Other datasets in `data_adapters/` are supported as expansion sources.

Do not claim the model detects every debris category — only those represented in the validated training data.

---

## Model Weights

Not committed to the repository. Place trained checkpoints at:
```
models/yolo11/weights/yolo11n_1c.pt
models/mobilenetv3/weights/mobilenetv3_verifier.pt
```

---

## Experimental / Future Modules

| Module | Location | Status |
|--------|----------|--------|
| RCDI-YOLO detector | `models/rcdi_yolo/` | Experimental — not validated for NEMO |
| Acoustic evidence fusion | `scoring/` | Rule-based scoring implemented |
| Hard-negative mining | `tools/mine_hard_negatives.py` | Infrastructure ready |
| Temporal consistency | — | Future work |
| TensorRT optimization | — | Future work |

See `docs/FUTURE_WORK.md` for the complete research roadmap.

---

## Commands

| Action | Command |
|--------|---------|
| Smoke test | `python tools/smoke_test.py` |
| Train detector | `ultralytics train model=yolo11n.pt data=...` |
| Train verifier | `python -m models.mobilenetv3.train --config configs/config.yaml` |
| Inference | `python main.py --mode recorded --config configs/config.yaml --mission path/` |
| Dashboard (API) | `python main.py --mode api --config configs/config.yaml` |
| Dashboard (Tester) | Open `dashboard/index.html` in browser |
| Benchmark | `python tools/benchmark.py --config configs/config.yaml` |
| Tests | `pytest tests/ -v` |

---

## Project Structure

```
├── configs/              # Configuration files (mvp.yaml, config.yaml, research_rcdi.yaml)
├── dashboard/            # NEMO REST API + dashboard UI
│   ├── app.py            # FastAPI server
│   └── index.html        # Dashboard frontend
├── database/             # MongoDB Atlas integration
├── models/
│   ├── yolo11/           # YOLO11n-1C validated MVP detector
│   ├── mobilenetv3/      # MobileNetV3-Small verifier
│   └── rcdi_yolo/        # RCDI-YOLO experimental (not default)
├── inference/            # Two-stage pipeline + types
├── preprocessing/        # Sonar image preprocessing
├── scoring/              # Evidence fusion scoring
├── metadata/             # GPS/ping alignment
├── reporting/            # JSON/CSV export
├── data_adapters/        # Dataset conversion adapters
├── tests/                # Unit and integration tests
├── tools/                # Smoke test, benchmark, utilities
├── docs/                 # Architecture, experiments, references
└── main.py               # Entry point
```

---

## Documentation

- `docs/EXPERIMENT_HANDOFF.md` — Authoritative experiment record
- `docs/EXPERIMENT_RESULTS.md` — Detailed T1–T4 results
- `docs/VALIDATED_MVP.md` — MVP architecture rationale
- `docs/FUTURE_WORK.md` — Research roadmap
- `docs/REQUIREMENT_ALIGNMENT.md` — Organizer requirement mapping
- `docs/architecture.md` — System design
- `docs/rcdi_reference.md` — RCDI-YOLO paper traceability
- `docs/dataset.md` — Dataset instructions

---

## Limitations

1. Detection limited to classes present in training data (KLSG aircraft + shipwreck)
2. Confidence scores are not statistically calibrated
3. No real-time streaming validation performed
4. RCDI-YOLO not experimentally compared against YOLO11n-1C on the same data
5. Evidence fusion is rule-based, not learned
6. Requires trained model weights for production inference

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

*NEMO — Built for Smart India Hackathon 2025*
