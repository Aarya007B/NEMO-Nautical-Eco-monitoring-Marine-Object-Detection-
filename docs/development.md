# NEMO Development Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/smoke_test.py
```

## Environment Variables

```bash
# MongoDB Atlas (optional — required for persistent storage)
export NEMO_MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
export NEMO_MONGO_DB="nemo"
```

## Testing

```bash
pytest tests/ -v
```

## Commands

| Action | Command |
|--------|---------|
| Smoke test | `python tools/smoke_test.py` |
| Verifier train | `python -m models.mobilenetv3.train --config configs/config.yaml` |
| Inference | `python main.py --mode recorded --config configs/config.yaml --mission path/` |
| Dashboard (API) | `python main.py --mode api --config configs/config.yaml` |
| Benchmark | `python tools/benchmark.py --config configs/config.yaml` |
| Split creation | `python tools/create_split.py --config configs/config.yaml` |

## Device Selection

Set in `configs/config.yaml`:

```yaml
runtime:
  device: auto  # auto | cuda | mps | cpu
```

## Configuration Files

| Config | Purpose |
|--------|---------|
| `configs/config.yaml` | Default MVP config (YOLO11n-1C) |
| `configs/mvp.yaml` | Explicit MVP config with experiment annotations |
| `configs/research_rcdi.yaml` | RCDI-YOLO research config (not default) |
| `configs/mobilenetv3.yaml` | MobileNetV3 verifier config |
