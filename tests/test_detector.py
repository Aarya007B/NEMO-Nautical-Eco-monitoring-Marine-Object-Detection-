"""
tests/test_detector.py

Tests for the full RCDI-YOLO model forward pass and detector interface.
All tests use random tensors — no trained weights or real data required.
"""
import pytest
import torch

from models.rcdi_yolo.model import RCDIYOLOModel
from inference.types import Detection, BoundingBox

DEVICE = torch.device("cpu")


@pytest.fixture
def small_model():
    """Tiny RCDI-YOLO model for fast CPU tests."""
    return RCDIYOLOModel(
        in_channels=1,
        num_classes=1,
        base_channels=[32, 64, 128, 128, 128],
        width_multiple=0.25,
        depth_multiple=0.33,
        reg_max=4,
        backbone_lan_positions=[1, 2],
        neck_lan_positions=[1, 2, 4],
        conf_threshold=0.0,
    ).to(DEVICE)


class TestRCDIYOLOModelForwardPass:
    def test_forward_640x640(self, small_model):
        preds = small_model(torch.randn(1, 1, 640, 640))
        assert isinstance(preds, list) and len(preds) == 3

    def test_forward_320x320(self, small_model):
        preds = small_model(torch.randn(1, 1, 320, 320))
        assert len(preds) == 3

    def test_output_channels(self, small_model):
        preds = small_model(torch.randn(1, 1, 640, 640))
        expected_ch = 4 * small_model.reg_max + small_model.num_classes
        for pred in preds:
            assert pred.shape[1] == expected_ch

    def test_output_spatial_hierarchy(self, small_model):
        preds = small_model(torch.randn(1, 1, 640, 640))
        assert preds[0].shape[2] > preds[1].shape[2] > preds[2].shape[2]

    def test_output_finite(self, small_model):
        preds = small_model(torch.randn(1, 1, 640, 640))
        for pred in preds:
            assert torch.isfinite(pred).all()

    def test_gradient_flows(self, small_model):
        x = torch.randn(1, 1, 160, 160, requires_grad=True)
        small_model.train()
        loss = sum(p.sum() for p in small_model(x))
        loss.backward()
        assert x.grad is not None


class TestRCDIYOLOModelPredict:
    def test_returns_list(self, small_model):
        dets = small_model.predict(torch.randn(1, 1, 640, 640), frame_id="frame_001")
        assert isinstance(dets, list)

    def test_detections_have_correct_type(self, small_model):
        for d in small_model.predict(torch.randn(1, 1, 640, 640)):
            assert isinstance(d, Detection)
            assert isinstance(d.bbox, BoundingBox)

    def test_confidence_filter_works(self):
        model = RCDIYOLOModel(
            in_channels=1, num_classes=1,
            base_channels=[32, 64, 64, 64, 64],
            width_multiple=0.25, depth_multiple=0.33,
            reg_max=4, conf_threshold=1.1,
        )
        assert len(model.predict(torch.randn(1, 1, 320, 320))) == 0

    def test_1ch_input_expected(self, small_model):
        with pytest.raises((RuntimeError, Exception)):
            small_model(torch.randn(1, 3, 640, 640))


class TestRCDIYOLOModelFromConfig:
    def test_from_config_loads(self):
        model = RCDIYOLOModel.from_config("configs/rcdi_yolo_1c.yaml")
        assert isinstance(model, RCDIYOLOModel)

    def test_from_config_forward_pass(self):
        model = RCDIYOLOModel.from_config("configs/rcdi_yolo_1c.yaml")
        preds = model(torch.randn(1, 1, 320, 320))
        assert len(preds) == 3
