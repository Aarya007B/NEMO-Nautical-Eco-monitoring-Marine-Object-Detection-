"""
tools/convert_dataset.py — Convert a raw dataset to NEMO canonical format.

Usage:
    python tools/convert_dataset.py --dataset seabedobjects --root data/raw/seabedobjects
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ADAPTERS = {
    "seabedobjects": "data_adapters.seabedobjects.SeabedObjectsAdapter",
    "kaggle_sss": "data_adapters.kaggle_sss.KaggleSSSAdapter",
    "opensonardatasets": "data_adapters.opensonardatasets.OpenSonarDatasetsAdapter",
    "dfki_uxo": "data_adapters.dfki_uxo.DFKIUXOAdapter",
    "uci_sonar": "data_adapters.uci_sonar.UCISonarAdapter",
}


def main():
    parser = argparse.ArgumentParser(description="Convert Dataset")
    parser.add_argument("--dataset", required=True, choices=list(ADAPTERS.keys()))
    parser.add_argument("--root", required=True, help="Path to raw dataset root")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    args = parser.parse_args()

    # Dynamic import
    module_path, class_name = ADAPTERS[args.dataset].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    AdapterClass = getattr(mod, class_name)

    adapter = AdapterClass(root=args.root)
    records = adapter.load()

    logger.info("Loaded %d records from %s", len(records), args.dataset)
    logger.info("Output directory: %s", args.output)

    # For now, just report — full conversion to YOLO format is Phase 4+
    for r in records[:5]:
        logger.info("  Sample: %s (class=%s, canonical=%s)",
                     r.frame_id, r.source_class, r.canonical_class)
    if len(records) > 5:
        logger.info("  ... and %d more", len(records) - 5)


if __name__ == "__main__":
    main()
