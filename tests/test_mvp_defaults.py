"""
tests/test_mvp_defaults.py

Integration tests verifying that the NEMO MVP defaults are correctly
configured: YOLO11n-1C as default detector, raw native preprocessing,
MobileNetV3 as verifier, and MongoDB integration.
"""
import pytest
import yaml


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMVPConfigDefaults:
    """Verify that default config.yaml uses validated MVP settings."""

    @pytest.fixture
    def config(self):
        with open("configs/config.yaml") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def mvp_config(self):
        with open("configs/mvp.yaml") as f:
            return yaml.safe_load(f)

    def test_default_detector_is_yolo11n(self, config):
        """Default detector must be YOLO11n-1C, NOT RCDI."""
        arch = config["detector"]["architecture"]
        assert arch == "yolo11n_1c", (
            f"Default detector is '{arch}', expected 'yolo11n_1c'. "
            "RCDI-YOLO must NOT be the default."
        )

    def test_default_detector_is_not_rcdi(self, config):
        """Explicitly verify RCDI is not the default."""
        arch = config["detector"]["architecture"]
        assert "rcdi" not in arch.lower(), (
            "RCDI must not be in the default detector architecture name."
        )

    def test_default_preprocessing_no_normalize(self, config):
        """MVP default: raw native input, no normalization."""
        assert config["preprocessing"]["normalize"] is False, (
            "MVP default preprocessing must have normalize=false (raw native input)."
        )

    def test_default_preprocessing_no_denoise(self, config):
        assert config["preprocessing"]["denoise"]["enabled"] is False

    def test_default_preprocessing_no_contrast(self, config):
        assert config["preprocessing"]["contrast"]["enabled"] is False

    def test_default_verifier_is_mobilenetv3(self, config):
        """Verifier must be MobileNetV3-Small."""
        arch = config["verifier"]["architecture"]
        assert arch == "mobilenet_v3_small"

    def test_project_name_is_nemo(self, config):
        """Project name must be NEMO, not SonarGuard."""
        name = config["project"]["name"]
        assert name == "NEMO", f"Project name is '{name}', expected 'NEMO'"
        assert "sonarguard" not in name.lower()

    def test_mvp_config_matches_default(self, config, mvp_config):
        """MVP config and default config must use same detector."""
        assert config["detector"]["architecture"] == mvp_config["detector"]["architecture"]

    def test_mvp_config_no_normalize(self, mvp_config):
        assert mvp_config["preprocessing"]["normalize"] is False


# ---------------------------------------------------------------------------
# Detector factory tests
# ---------------------------------------------------------------------------

class TestDetectorFactory:
    """Verify the detector factory selects the correct model."""

    def test_yolo11n_factory(self):
        """Factory should produce YOLO11nDetector for yolo11n_1c config."""
        from inference.detector import _build_detector_model
        import torch
        model = _build_detector_model(
            {"architecture": "yolo11n_1c"},
            torch.device("cpu"),
        )
        from models.yolo11.model import YOLO11nDetector
        assert isinstance(model, YOLO11nDetector)

    def test_rcdi_factory(self):
        """Factory should produce RCDIYOLOModel for rcdi_yolov8_1c config."""
        from inference.detector import _build_detector_model
        import torch
        model = _build_detector_model(
            {"architecture": "rcdi_yolov8_1c", "config": "configs/rcdi_yolo_1c.yaml"},
            torch.device("cpu"),
        )
        from models.rcdi_yolo.model import RCDIYOLOModel
        assert isinstance(model, RCDIYOLOModel)


# ---------------------------------------------------------------------------
# YOLO11n model tests
# ---------------------------------------------------------------------------

class TestYOLO11nDetector:
    """Test YOLO11n-1C model in demo mode (no trained weights)."""

    def test_demo_mode_returns_empty(self):
        """In demo mode without weights, predict returns empty list."""
        import torch
        from models.yolo11.model import YOLO11nDetector
        model = YOLO11nDetector(
            weights_path="nonexistent.pt",
            demo_mode=True,
        )
        x = torch.randn(1, 1, 640, 640)
        dets = model.predict(x, frame_id="test")
        assert isinstance(dets, list)
        assert len(dets) == 0

    def test_from_config(self):
        """from_config should construct a valid YOLO11nDetector."""
        from models.yolo11.model import YOLO11nDetector
        cfg = {
            "architecture": "yolo11n_1c",
            "weights": "nonexistent.pt",
            "num_classes": 1,
            "confidence_threshold": 0.25,
            "nms_iou": 0.70,
            "demo_mode": True,
        }
        model = YOLO11nDetector.from_config(cfg)
        assert isinstance(model, YOLO11nDetector)
        assert model.num_classes == 1
        assert model.conf_threshold == 0.25

    def test_forward_raises(self):
        """forward() should raise NotImplementedError for ultralytics models."""
        import torch
        from models.yolo11.model import YOLO11nDetector
        model = YOLO11nDetector(weights_path="nonexistent.pt", demo_mode=True)
        with pytest.raises(NotImplementedError):
            model(torch.randn(1, 1, 640, 640))


# ---------------------------------------------------------------------------
# MongoDB integration tests
# ---------------------------------------------------------------------------

class TestDatabaseIntegration:
    """Test database module without requiring MongoDB connection."""

    def test_import_connection(self):
        """database.connection module should be importable."""
        from database.connection import get_database, is_connected
        # Without NEMO_MONGO_URI set, should return None/False
        assert is_connected() is False

    def test_import_operations(self):
        """database.operations module should be importable."""
        from database.operations import store_mission, list_missions
        # Without MongoDB, store should return None
        result = store_mission("test", "/tmp", 0, 0)
        assert result is None

    def test_list_missions_empty(self):
        """Without MongoDB, list_missions should return empty list."""
        from database.operations import list_missions
        missions = list_missions()
        assert missions == []


# ---------------------------------------------------------------------------
# Branding tests
# ---------------------------------------------------------------------------

class TestBranding:
    """Verify project branding is NEMO, not SonarGuard."""

    def test_config_name(self):
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["project"]["name"] == "NEMO"

    def test_readme_title(self):
        with open("README.md") as f:
            first_line = f.readline().strip()
        assert "NEMO" in first_line
        assert "SonarGuard" not in first_line
