# NEMO Dataset Instructions

## Supported Datasets

| Dataset | Type | Adapter |
|---------|------|---------|
| SeabedObjects Ship-and-Airplane | Sonar images + YOLO labels | `seabedobjects.py` |
| Kaggle Side-Scan Sonar | Sonar images + labels | `kaggle_sss.py` |
| REMARO OpenSonarDatasets | Sonar images | `opensonardatasets.py` |
| DFKI-RIC UXO 2024 | Sonar images + annotations | `dfki_uxo.py` |
| UCI Sonar Mines vs Rocks | Feature vectors (no images) | `uci_sonar.py` |
| AI4Shipwrecks | Side-scan strips + masks (shipwreck) | convert via `tools/convert_ai4shipwrecks.py` |

## Dataset Scope

The validated MVP experiments (T1–T4) were performed on the **KLSG / SeabedObjects** dataset. Other datasets are supported as expansion sources but were not used in the validated MVP training.

Do not claim the model detects every debris category — only those represented in the validated training data (aircraft, shipwreck targets).

## Setup

Place raw datasets in `data/raw/<dataset_name>/`. Do NOT modify originals.

## Conversion

```bash
python tools/convert_dataset.py --dataset seabedobjects --root data/raw/seabedobjects
```

## AI4Shipwrecks (shipwreck expansion)

1. Download the dataset from https://umfieldrobotics.github.io/ai4shipwrecks/
   (expects `{train,test}/images`, `{train,test}/labels`, `extras/terrain/`).
2. Convert strips + masks to NEMO YOLO format (class 3 = shipwreck):

```bash
python tools/convert_ai4shipwrecks.py --src ~/Downloads/AI4Shipwrecks
```

Output lands in `data/raw/ai4shipwrecks/{train,valid}/{images,labels}/` (git-ignored,
regenerate locally — do not commit). Training configs live in
`data/raw/combined_nemo_ai4/` (`data.yaml` for local runs, `data_colab.yaml` for Colab).

## Splits

```bash
python tools/create_split.py --config configs/config.yaml
```

Always use mission-grouped splitting to avoid data leakage.
