# SonarGuard Development Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/smoke_test.py
```

## Testing

```bash
pytest tests/ -v
```

## Commands

| Action | Command |
|--------|---------|
| Smoke test | `python tools/smoke_test.py` |
| Detector train | `python -m models.rcdi_yolo.train --config configs/config.yaml` |
| Verifier train | `python -m models.mobilenetv3.train --config configs/config.yaml` |
| Inference | `python main.py --mode recorded --config configs/config.yaml --mission path/` |
| Dashboard | `streamlit run dashboard/app.py` |
| Benchmark | `python tools/benchmark.py --config configs/config.yaml` |
| Split creation | `python tools/create_split.py --config configs/config.yaml` |
| Hard negatives | `python tools/mine_hard_negatives.py --config configs/config.yaml` |

## Device Selection

Set in `configs/config.yaml`:

```yaml
runtime:
  device: auto  # auto | cuda | mps | cpu
```
