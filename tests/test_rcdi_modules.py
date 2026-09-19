"""
tests/test_rcdi_modules.py

Unit tests for RCDI-YOLO custom modules.
All tests use random tensors — no trained weights or real data required.
"""
import pytest
import torch
import numpy as np
import cv2

from models.rcdi_yolo.modules.lanconvnextv2 import (
    LANConvNeXtv2, MultiScaleDilatedConv, ChannelAttention, ConvBnSilu,
)
from models.rcdi_yolo.modules.dysample import DySample
from models.rcdi_yolo.modules.implicit import ImplicitA, ImplicitM
from models.rcdi_yolo.modules.implicit_head import ImplicitHead, HeadBranch, DFL

DEVICE = torch.device("cpu")


class TestConvBnSilu:
    def test_shape(self):
        m = ConvBnSilu(32, 64, k=3).to(DEVICE)
        y = m(torch.randn(2, 32, 40, 40))
        assert y.shape == (2, 64, 40, 40)

    def test_output_finite(self):
        m = ConvBnSilu(16, 32).to(DEVICE)
        assert torch.isfinite(m(torch.randn(1, 16, 20, 20))).all()


class TestMultiScaleDilatedConv:
    def test_shape_preserved(self):
        m = MultiScaleDilatedConv(64).to(DEVICE)
        x = torch.randn(2, 64, 32, 32)
        assert m(x).shape == x.shape

    def test_output_finite(self):
        m = MultiScaleDilatedConv(32).to(DEVICE)
        assert torch.isfinite(m(torch.randn(1, 32, 16, 16))).all()


class TestChannelAttention:
    def test_shape_preserved(self):
        m = ChannelAttention(64).to(DEVICE)
        x = torch.randn(2, 64, 32, 32)
        assert m(x).shape == x.shape


class TestLANConvNeXtv2:
    @pytest.mark.parametrize("in_ch,out_ch,h,w", [
        (32, 32, 40, 40),
        (64, 64, 20, 20),
        (128, 256, 10, 10),
        (256, 128, 5, 5),
    ])
    def test_shape(self, in_ch, out_ch, h, w):
        m = LANConvNeXtv2(in_ch, out_ch).to(DEVICE)
        y = m(torch.randn(2, in_ch, h, w))
        assert y.shape == (2, out_ch, h, w)

    def test_output_finite(self):
        m = LANConvNeXtv2(64, 64).to(DEVICE)
        assert torch.isfinite(m(torch.randn(1, 64, 32, 32))).all()

    def test_residual_same_channels(self):
        m = LANConvNeXtv2(64, 64).to(DEVICE)
        assert isinstance(m.shortcut, torch.nn.Identity)

    def test_shortcut_different_channels(self):
        m = LANConvNeXtv2(32, 64).to(DEVICE)
        assert not isinstance(m.shortcut, torch.nn.Identity)

    def test_gradient_flows(self):
        m = LANConvNeXtv2(32, 32).to(DEVICE)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestDySample:
    @pytest.mark.parametrize("in_ch,scale", [(32, 2), (64, 2), (128, 2)])
    def test_upsamples_correctly(self, in_ch, scale):
        m = DySample(in_ch, scale=scale).to(DEVICE)
        x = torch.randn(1, in_ch, 20, 20)
        y = m(x)
        assert y.shape == (1, in_ch, 20 * scale, 20 * scale)

    def test_output_finite(self):
        m = DySample(64, scale=2).to(DEVICE)
        assert torch.isfinite(m(torch.randn(2, 64, 10, 10))).all()

    def test_gradient_flows(self):
        m = DySample(32, scale=2).to(DEVICE)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestImplicit:
    def test_implicit_a_shape(self):
        m = ImplicitA(64).to(DEVICE)
        x = torch.randn(2, 64, 20, 20)
        assert m(x).shape == x.shape

    def test_implicit_m_shape(self):
        m = ImplicitM(64).to(DEVICE)
        x = torch.randn(2, 64, 20, 20)
        assert m(x).shape == x.shape

    def test_implicit_a_init_near_zero(self):
        m = ImplicitA(256).to(DEVICE)
        assert m.implicit.abs().mean().item() < 0.1

    def test_implicit_m_init_near_one(self):
        m = ImplicitM(256).to(DEVICE)
        assert abs(m.implicit.mean().item() - 1.0) < 0.1

    def test_both_have_gradients(self):
        ia = ImplicitA(32).to(DEVICE)
        im = ImplicitM(32).to(DEVICE)
        x = torch.randn(1, 32, 8, 8)
        im(ia(x)).sum().backward()
        assert ia.implicit.grad is not None
        assert im.implicit.grad is not None


class TestDFL:
    def test_output_shape(self):
        m = DFL(reg_max=16).to(DEVICE)
        y = m(torch.randn(1, 4 * 16, 20, 20))
        assert y.shape == (1, 4, 20, 20)

    def test_non_trainable(self):
        m = DFL(reg_max=16).to(DEVICE)
        assert not m.conv.weight.requires_grad


class TestImplicitHead:
    def test_forward_shape(self):
        head = ImplicitHead(in_channels=[64, 128, 256], num_classes=1, reg_max=16)
        feats = [
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 256, 20, 20),
        ]
        outputs = head(feats)
        assert len(outputs) == 3
        expected_ch = 4 * 16 + 1
        assert outputs[0].shape == (1, expected_ch, 80, 80)
        assert outputs[1].shape == (1, expected_ch, 40, 40)
        assert outputs[2].shape == (1, expected_ch, 20, 20)

    def test_output_finite(self):
        head = ImplicitHead(in_channels=[32, 64, 128], num_classes=1, reg_max=8)
        feats = [
            torch.randn(1, 32, 40, 40),
            torch.randn(1, 64, 20, 20),
            torch.randn(1, 128, 10, 10),
        ]
        for out in head(feats):
            assert torch.isfinite(out).all()


class TestRCDILoss:
    def test_loss_computes(self):
        from models.rcdi_yolo.losses import RCDILoss
        loss_fn = RCDILoss(num_classes=1, reg_max=16)
        preds = [
            torch.randn(2, 4*16+1, 80, 80),
            torch.randn(2, 4*16+1, 40, 40),
            torch.randn(2, 4*16+1, 20, 20),
        ]
        total, d = loss_fn(preds)
        assert torch.isfinite(total).item()
        assert "box" in d and "cls" in d and "dfl" in d

    def test_loss_no_targets(self):
        from models.rcdi_yolo.losses import RCDILoss
        loss_fn = RCDILoss(num_classes=1, reg_max=16)
        preds = [torch.randn(1, 4*16+1, 80, 80)]
        total, d = loss_fn(preds)
        assert torch.isfinite(total).item()


class TestSonarDetectionDataset:
    def test_empty_dataset(self):
        from models.rcdi_yolo.dataset import SonarDetectionDataset
        ds = SonarDetectionDataset([], [], image_size=640, num_classes=1)
        assert len(ds) == 0

    def test_dataset_constructs(self):
        import tempfile, os
        from models.rcdi_yolo.dataset import SonarDetectionDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            import cv2
            cv2.imwrite(img_path, (np.random.randint(0, 255, (640, 640), dtype=np.uint8)))
            ds = SonarDetectionDataset([img_path], [""], image_size=640, num_classes=1)
            assert len(ds) == 1
