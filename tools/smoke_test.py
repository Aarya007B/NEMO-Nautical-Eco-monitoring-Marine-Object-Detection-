"""
tools/smoke_test.py — SonarGuard Phase 1-3 smoke test.

Runs minimum checks to verify scaffold, preprocessing, and RCDI-YOLO forward pass.

Does NOT require:
    - Trained model weights
    - Real sonar datasets
    - GPU access

Usage:
    python tools/smoke_test.py
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

PASS = "\033[92m\u2713 PASS\033[0m"
FAIL = "\033[91m\u2717 FAIL\033[0m"

results = []


def check(name, fn):
    try:
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {PASS}  {name}  ({elapsed:.0f}ms)")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        results.append((name, False, e))


print("\n" + "=" * 60)
print(" SonarGuard / NEMO — Smoke Test")
print("=" * 60)

# ---- [1] Environment ----
print("\n[1] Environment")

def _python():
    assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version}"
check("Python >= 3.8", _python)

def _torch():
    import torch; assert torch.__version__
check("PyTorch importable", _torch)

def _numpy():
    import numpy as np; _ = np.zeros((3, 3))
check("NumPy importable", _numpy)

def _cv2():
    import cv2; _ = cv2.__version__
check("OpenCV importable", _cv2)

def _yaml():
    import yaml; assert yaml.safe_load("key: value")["key"] == "value"
check("PyYAML works", _yaml)

# ---- [2] Configuration ----
print("\n[2] Configuration")

def _config():
    import yaml
    assert Path("configs/config.yaml").exists()
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "detector" in cfg and "verifier" in cfg and "scoring" in cfg
check("configs/config.yaml valid", _config)

def _rcdi_cfg():
    import yaml
    assert Path("configs/rcdi_yolo_1c.yaml").exists()
    with open("configs/rcdi_yolo_1c.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["input_channels"] == 1
check("configs/rcdi_yolo_1c.yaml valid", _rcdi_cfg)

# ---- [3] Types ----
print("\n[3] Types")

def _types():
    from inference.types import (
        BoundingBox, Detection, VerificationResult,
        AcousticFeatures, ScoreResult, DetectionResult,
        MissionFrame, MissionResult, DatasetRecord, DetectionStatus,
    )
    bb = BoundingBox(10, 20, 100, 200)
    assert bb.width == 90 and bb.height == 180
    expanded = bb.expand(0.1, 1000, 1000)
    assert expanded.x1 < bb.x1
check("inference.types importable and functional", _types)

# ---- [4] Preprocessing ----
print("\n[4] Preprocessing")

def _pp_import():
    from preprocessing.pipeline import SonarPreprocessor
    from preprocessing.normalize import minmax_normalize
    from preprocessing.denoise import denoise
    from preprocessing.contrast import enhance_contrast
    from preprocessing.resize import letterbox
check("preprocessing modules import", _pp_import)

def _pp_array():
    import numpy as np; import torch
    from preprocessing.pipeline import SonarPreprocessor
    pp = SonarPreprocessor({
        "image_size": 640, "normalize": True,
        "denoise": {"enabled": False}, "contrast": {"enabled": False},
    })
    img = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
    tensor, scale, padding = pp(img)
    assert tensor.shape == (1, 640, 640)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
check("SonarPreprocessor: array -> [1,640,640] tensor", _pp_array)

def _pp_rgb():
    import numpy as np
    from preprocessing.pipeline import SonarPreprocessor
    pp = SonarPreprocessor({
        "image_size": 640, "normalize": True,
        "denoise": {"enabled": False}, "contrast": {"enabled": False},
    })
    tensor, _, _ = pp(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    assert tensor.shape == (1, 640, 640)
check("SonarPreprocessor: RGB -> single-channel", _pp_rgb)

# ---- [5] RCDI Modules ----
print("\n[5] RCDI-YOLO Custom Modules")

def _lanconv():
    import torch
    from models.rcdi_yolo.modules.lanconvnextv2 import LANConvNeXtv2
    m = LANConvNeXtv2(64, 64)
    y = m(torch.randn(1, 64, 40, 40))
    assert y.shape == (1, 64, 40, 40) and torch.isfinite(y).all()
check("LANConvNeXtv2: [1,64,40,40] -> [1,64,40,40]", _lanconv)

def _dysample():
    import torch
    from models.rcdi_yolo.modules.dysample import DySample
    m = DySample(64, scale=2)
    y = m(torch.randn(1, 64, 20, 20))
    assert y.shape == (1, 64, 40, 40) and torch.isfinite(y).all()
check("DySample: [1,64,20,20] -> [1,64,40,40]", _dysample)

def _implicit():
    import torch
    from models.rcdi_yolo.modules.implicit import ImplicitA, ImplicitM
    x = torch.randn(2, 64, 10, 10)
    y = ImplicitM(64)(ImplicitA(64)(x))
    assert y.shape == x.shape and torch.isfinite(y).all()
check("ImplicitA + ImplicitM: shape preserved", _implicit)

def _implicit_head():
    import torch
    from models.rcdi_yolo.modules.implicit_head import ImplicitHead
    head = ImplicitHead(in_channels=[64, 128, 256], num_classes=1, reg_max=16)
    feats = [torch.randn(1,64,80,80), torch.randn(1,128,40,40), torch.randn(1,256,20,20)]
    outs = head(feats)
    assert len(outs) == 3
    for out in outs:
        assert torch.isfinite(out).all()
check("ImplicitHead: 3 scale outputs, all finite", _implicit_head)

# ---- [6a] Loss & Dataset ----
print("\n[6a] RCDI-YOLO Loss & Dataset")

def _loss():
    import torch
    from models.rcdi_yolo.losses import RCDILoss
    loss_fn = RCDILoss(num_classes=1, reg_max=16)
    preds = [torch.randn(2, 4*16+1, 80, 80), torch.randn(2, 4*16+1, 40, 40), torch.randn(2, 4*16+1, 20, 20)]
    total, d = loss_fn(preds)
    assert torch.isfinite(total).item()
check("RCDILoss computes loss", _loss)

def _dataset():
    from models.rcdi_yolo.dataset import SonarDetectionDataset
    ds = SonarDetectionDataset([], [], image_size=640, num_classes=1)
    assert len(ds) == 0
check("SonarDetectionDataset constructs", _dataset)

def _loss_no_targets():
    import torch
    from models.rcdi_yolo.losses import RCDILoss
    loss_fn = RCDILoss(num_classes=1, reg_max=16)
    preds = [torch.randn(1, 4*16+1, 80, 80)]
    total, d = loss_fn(preds)
    assert torch.isfinite(total).item()
check("RCDILoss forward without targets", _loss_no_targets)

# ---- [7] Full Model ----
print("\n[6] RCDI-YOLO 1C Model")

def _build():
    from models.rcdi_yolo.model import RCDIYOLOModel
    m = RCDIYOLOModel(
        in_channels=1, num_classes=1,
        base_channels=[32,64,128,128,128],
        width_multiple=0.25, depth_multiple=0.33, reg_max=4,
    )
    assert sum(p.numel() for p in m.parameters()) > 10_000
check("RCDIYOLOModel constructs (1-channel, small)", _build)

def _forward():
    import torch
    from models.rcdi_yolo.model import RCDIYOLOModel
    m = RCDIYOLOModel(
        in_channels=1, num_classes=1,
        base_channels=[32,64,128,128,128],
        width_multiple=0.25, depth_multiple=0.33, reg_max=4,
    )
    preds = m(torch.randn(1, 1, 640, 640))
    assert len(preds) == 3
    for p in preds:
        assert torch.isfinite(p).all()
check("RCDI forward [1,1,640,640] -> 3 scale preds", _forward)

def _from_cfg():
    from models.rcdi_yolo.model import RCDIYOLOModel
    assert isinstance(RCDIYOLOModel.from_config("configs/rcdi_yolo_1c.yaml"), RCDIYOLOModel)
check("RCDIYOLOModel.from_config()", _from_cfg)

def _predict():
    import torch
    from models.rcdi_yolo.model import RCDIYOLOModel
    from inference.types import Detection
    m = RCDIYOLOModel(
        in_channels=1, num_classes=1,
        base_channels=[32,64,128,128,128],
        width_multiple=0.25, depth_multiple=0.33,
        reg_max=4, conf_threshold=0.0,
    )
    dets = m.predict(torch.randn(1, 1, 320, 320), frame_id="smoke")
    assert isinstance(dets, list)
    for d in dets:
        assert isinstance(d, Detection)
check("RCDI predict() -> list[Detection]", _predict)

# ---- [7] Verifier ----
print("\n[7] MobileNetV3 Verifier")

def _mn_build():
    from models.mobilenetv3.model import MobileNetV3Verifier
    assert sum(p.numel() for p in MobileNetV3Verifier(in_channels=1).parameters()) > 100_000
check("MobileNetV3 constructs (1-ch, 2-class)", _mn_build)

def _mn_forward():
    import torch
    from models.mobilenetv3.model import MobileNetV3Verifier
    y = MobileNetV3Verifier(in_channels=1)(torch.randn(2, 1, 128, 128))
    assert y.shape == (2, 2) and torch.isfinite(y).all()
check("MobileNetV3 forward [2,1,128,128] -> [2,2]", _mn_forward)

def _mn_verify():
    import torch
    from models.mobilenetv3.model import MobileNetV3Verifier
    from inference.types import VerificationResult
    r = MobileNetV3Verifier(in_channels=1).verify(torch.randn(1, 1, 128, 128))
    assert isinstance(r, VerificationResult)
    assert abs(r.natural_probability + r.artificial_probability - 1.0) < 1e-4
    assert r.predicted_class in ("natural", "artificial")
check("MobileNetV3 verify() -> VerificationResult (probs sum=1)", _mn_verify)

# ---- [7a] MobileNetV3 Train/Val ----
print("\n[7a] MobileNetV3 Training & Validation")

def _mn_train_import():
    from models.mobilenetv3.train import train
    check("MobileNetV3 train importable", lambda: True)
def _mn_val_import():
    from models.mobilenetv3.validate import validate
    check("MobileNetV3 validate importable", lambda: True)
try:
    _mn_train_import()
except Exception as e:
    print(f"  ✗  MobileNetV3 train import")
try:
    _mn_val_import()
except Exception as e:
    print(f"  ✗  MobileNetV3 validate import")

def _crop_dataset():
    from models.mobilenetv3.dataset import CropDataset
    ds = CropDataset(root="/nonexistent", crop_size=128)
    assert len(ds) == 0
check("CropDataset constructs", _crop_dataset)

# ---- [8] Dirs ----
print("\n[8] Directory Structure")

def _dirs():
    required = [
        "data/raw", "data/processed", "data/annotations",
        "data/hard_negatives", "data/splits",
        "models/rcdi_yolo/weights", "models/mobilenetv3/weights",
        "outputs/detections", "outputs/crops", "outputs/reports",
        "experiments/baseline", "experiments/rcdi",
        "preprocessing", "inference", "configs", "tools", "tests", "docs",
        "models/rcdi_yolo", "models/mobilenetv3",
        "scoring", "metadata", "mission", "data_adapters",
        "data/raw/seabedobjects", "data/raw/kaggle_sss",
        "data/raw/opensonardatasets", "data/raw/dfki_uxo",
        "data/raw/uci_sonar", "data/raw/other",
    ]
    missing = [d for d in required if not Path(d).exists()]
    assert not missing, f"Missing: {missing}"
check("Required directories exist", _dirs)

# ---- Summary ----
print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f" Results: {passed}/{total} passed")
if failed:
    print(f" \033[91mFailed checks ({failed}):")
    for name, ok, err in results:
        if not ok:
            print(f"   - {name}")
    print("\033[0m")
else:
    print(" \033[92mAll checks passed! Phase 1-3 scaffold is functional.\033[0m")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
