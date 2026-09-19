# SonarGuard / NEMO 🌊
### AI-Powered Automated Underwater Marine Debris & Anomaly Detection System

**NEMO** is a modular, two-stage AI pipeline that ingests side-scan sonar imagery, detects man-made marine debris and anomalies, and presents findings through the NEMO mission dashboard.

> **Do not treat every sonar detection as a confirmed object.** NEMO uses a two-stage pipeline to reduce false positives caused by natural seabed structures.

---

## Architecture

```
Side-Scan Sonar Image
        |
        v
   Preprocessing
   (grayscale -> normalize -> resize -> [1,640,640])
        |
        v
   RCDI-YOLO 1C              <-- Stage 1: Candidate Detection
   (LANConvNeXtv2 + DySample + ImplicitHead)
        |
        v
   Candidate Bounding Boxes
        |
        v
   Crop Extraction
        |
        v
   MobileNetV3-Small         <-- Stage 2: Candidate Verification
   (natural vs artificial)
        |
        v
   Evidence Fusion
        |
        v
   artificialness_score + priority_score
        |
        v
   GPS / Ping Metadata
        |
       / \
      /   \
     v     v
  NEMO     JSON / CSV
Dashboard   Report
```

---

## RCDI-YOLO Detector

Based on: **Zhang, J. and Gao, B. (2025). "RCDI-YOLO: a target-detection method for complex environment side-scan sonar images based on improved YOLOv8." Frontiers in Marine Science 12:1679077.**

Three modifications to YOLOv8:

- **LANConvNeXtv2** — Multi-scale dilated convolution + channel attention (Backbone pos 1,2 + Neck pos 1,2,4)
- **DySample** — Dynamic learnable upsampling in FPN neck
- **ImplicitHead** — Detection head with ImplicitA/M adapters and DFL

> **SonarGuard Adaptation:** Input changed from 3-channel RGB to 1-channel sonar `[B, 1, H, W]`. See `docs/rcdi_reference.md`.

---

## Two-Stage Pipeline

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | RCDI-YOLO 1C | High recall — find all candidates |
| 2 | MobileNetV3-Small | Reduce false positives — artificial vs natural |

Scores stored independently: `detector_confidence`, `verifier_artificial_prob`, `artificialness_score`, `priority_score`.

---

## Getting Started

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Smoke Test (no GPU or data required)
```bash
python tools/smoke_test.py
```

### 3. Run Tests
```bash
pytest tests/ -v
```

---

## Datasets

Place raw datasets in `data/raw/<name>/`. See `docs/dataset.md`.

```bash
python tools/validate_dataset.py
python tools/convert_dataset.py --dataset seabedobjects --root data/raw/seabedobjects
python tools/create_split.py --config configs/config.yaml
```

---

## Model Weights

Not committed. Place in:
```
models/rcdi_yolo/weights/rcdi_yolo_1c.pt
models/mobilenetv3/weights/mobilenetv3_verifier.pt
```

---

## Commands

| Action | Command |
|--------|---------|
| Smoke test | `python tools/smoke_test.py` |
| Train detector | `python -m models.rcdi_yolo.train --config configs/config.yaml` |
| Train verifier | `python -m models.mobilenetv3.train --config configs/config.yaml` |
| Inference | `python main.py --mode recorded --config configs/config.yaml --mission path/` |
| Dashboard | `streamlit run dashboard/app.py` |
| Benchmark | `python tools/benchmark.py --config configs/config.yaml` |
| Hard negatives | `python tools/mine_hard_negatives.py --config configs/config.yaml` |

---

## Project Structure

```
├── configs/              # Configuration files
├── data/                 # Datasets, annotations, splits
├── preprocessing/        # Sonar image preprocessing
├── models/
│   ├── rcdi_yolo/        # RCDI-YOLO 1C detector
│   │   └── modules/      # LANConvNeXtv2, DySample, ImplicitHead
│   └── mobilenetv3/      # MobileNetV3-Small verifier
├── data_adapters/        # Dataset conversion adapters
├── inference/            # Two-stage pipeline + types
├── scoring/              # Evidence fusion scoring
├── metadata/             # GPS/ping alignment
├── mission/              # Mission frame sources
├── reporting/            # JSON/CSV export
├── dashboard/            # NEMO Streamlit dashboard
├── experiments/          # Experiment configurations
├── tools/                # Smoke test, benchmark, split tools
├── tests/                # Unit tests
├── docs/                 # Architecture, dataset, reference docs
└── outputs/              # Detection results, crops, reports
```

---

## Hardware

| Device | Support |
|--------|---------|
| NVIDIA GPU (CUDA) | Full training + inference |
| Apple M-series (MPS) | Inference + light training |
| CPU | Inference, tests, development |

---

## Documentation

- `docs/architecture.md` — Design rationale
- `docs/rcdi_reference.md` — Paper traceability
- `docs/dataset.md` — Dataset instructions
- `docs/development.md` — Development workflow

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

*SonarGuard / NEMO — Built for Smart India Hackathon 2025*
