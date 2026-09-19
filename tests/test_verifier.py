"""
tests/test_verifier.py

Tests for MobileNetV3-Small verifier.
"""
import pytest
import torch

from models.mobilenetv3.model import MobileNetV3Verifier
from inference.types import VerificationResult


class TestMobileNetV3Verifier:
    def test_build(self):
        m = MobileNetV3Verifier(in_channels=1, num_classes=2)
        assert sum(p.numel() for p in m.parameters()) > 100_000

    def test_forward_shape(self):
        m = MobileNetV3Verifier(in_channels=1, num_classes=2)
        y = m(torch.randn(2, 1, 128, 128))
        assert y.shape == (2, 2)
        assert torch.isfinite(y).all()

    def test_verify_returns_result(self):
        m = MobileNetV3Verifier(in_channels=1, num_classes=2)
        result = m.verify(torch.randn(1, 1, 128, 128))
        assert isinstance(result, VerificationResult)
        assert result.predicted_class in ("natural", "artificial")

    def test_verify_probs_sum_to_one(self):
        m = MobileNetV3Verifier(in_channels=1, num_classes=2)
        result = m.verify(torch.randn(1, 1, 128, 128))
        total = result.natural_probability + result.artificial_probability
        assert abs(total - 1.0) < 1e-4

    def test_3ch_rejected(self):
        m = MobileNetV3Verifier(in_channels=1, num_classes=2)
        with pytest.raises((RuntimeError, Exception)):
            m(torch.randn(1, 3, 128, 128))
