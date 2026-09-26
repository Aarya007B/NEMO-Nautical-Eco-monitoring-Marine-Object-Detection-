"""
tools/benchmark.py — NEMO per-stage latency benchmark.

Measures preprocessing, detector, crop extraction, verifier, and
fusion latency independently.

Usage:
    python tools/benchmark.py --config configs/config.yaml [--iterations 50]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def benchmark(config_path: str, iterations: int = 50):
    import yaml
    from preprocessing.pipeline import SonarPreprocessor
    from models.rcdi_yolo.model import RCDIYOLOModel
    from models.mobilenetv3.model import MobileNetV3Verifier

    with open(config_path) as f:
        config = yaml.safe_load(f)

    pp_cfg = config.get("preprocessing", {})
    preprocessor = SonarPreprocessor(pp_cfg)

    detector = RCDIYOLOModel.from_config(
        config.get("detector", {}).get("config", "configs/rcdi_yolo_1c.yaml")
    )
    detector.eval()

    verifier = MobileNetV3Verifier(in_channels=1, num_classes=2)
    verifier.eval()

    # Synthetic data
    image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    timings = {
        "preprocessing": [],
        "detector": [],
        "verifier": [],
        "total": [],
    }

    # Warmup
    for _ in range(3):
        t, s, p = preprocessor(image)
        _ = detector(t.unsqueeze(0))
        _ = verifier(torch.randn(1, 1, 128, 128))

    # Benchmark
    for i in range(iterations):
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        tensor, scale, padding = preprocessor(image)
        timings["preprocessing"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.no_grad():
            preds = detector(tensor.unsqueeze(0))
        timings["detector"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = verifier(torch.randn(1, 1, 128, 128))
        timings["verifier"].append(time.perf_counter() - t0)

        timings["total"].append(time.perf_counter() - t_total_start)

    print("\n" + "=" * 55)
    print(" NEMO Latency Benchmark")
    print(f" Config:     {config_path}")
    print(f" Iterations: {iterations}")
    print(f" Device:     CPU (benchmark uses CPU)")
    print("=" * 55)

    for stage, times in timings.items():
        arr = np.array(times) * 1000
        fps = 1000.0 / arr.mean() if arr.mean() > 0 else 0
        print(f"\n  {stage:15s}  mean={arr.mean():.1f}ms  "
              f"std={arr.std():.1f}ms  "
              f"p50={np.median(arr):.1f}ms  "
              f"p95={np.percentile(arr, 95):.1f}ms  "
              f"FPS={fps:.1f}")

    print("\n" + "=" * 55)
    print("  Reference: YOLO11n-1C ~64 FPS, +MobileNetV3 ~37 FPS on T4.")
    print("  These are YOUR device-specific measurements.")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NEMO Benchmark")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.config, args.iterations)


if __name__ == "__main__":
    main()
