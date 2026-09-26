# NEMO Dataset Instructions

## Supported Datasets

| Dataset | Type | Adapter |
|---------|------|---------|
| SeabedObjects Ship-and-Airplane | Sonar images + YOLO labels | `seabedobjects.py` |
| Kaggle Side-Scan Sonar | Sonar images + labels | `kaggle_sss.py` |
| REMARO OpenSonarDatasets | Sonar images | `opensonardatasets.py` |
| DFKI-RIC UXO 2024 | Sonar images + annotations | `dfki_uxo.py` |
| UCI Sonar Mines vs Rocks | Feature vectors (no images) | `uci_sonar.py` |

## Dataset Scope

The validated MVP experiments (T1–T4) were performed on the **KLSG / SeabedObjects** dataset. Other datasets are supported as expansion sources but were not used in the validated MVP training.

Do not claim the model detects every debris category — only those represented in the validated training data (aircraft, shipwreck targets).

## Setup

Place raw datasets in `data/raw/<dataset_name>/`. Do NOT modify originals.

## Conversion

```bash
python tools/convert_dataset.py --dataset seabedobjects --root data/raw/seabedobjects
```

## Splits

```bash
python tools/create_split.py --config configs/config.yaml
```

Always use mission-grouped splitting to avoid data leakage.
