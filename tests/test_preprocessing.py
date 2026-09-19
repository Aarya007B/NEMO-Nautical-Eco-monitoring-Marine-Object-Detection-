"""
tests/test_preprocessing.py

Tests for the sonar preprocessing pipeline.
All tests use synthetic random data — no real datasets required.
"""
import numpy as np
import pytest
import torch

from preprocessing.normalize import minmax_normalize, zscore_normalize, to_float32
from preprocessing.denoise import denoise
from preprocessing.contrast import enhance_contrast
from preprocessing.resize import letterbox, unletterbox_bbox
from preprocessing.pipeline import SonarPreprocessor


@pytest.fixture
def random_gray_u8():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(256, 320), dtype=np.uint8)


@pytest.fixture
def random_gray_f32():
    rng = np.random.default_rng(42)
    return rng.random(size=(256, 320)).astype(np.float32)


@pytest.fixture
def minimal_config():
    return {
        "image_size": 640,
        "normalize": True,
        "denoise": {"enabled": False},
        "contrast": {"enabled": False},
    }


class TestNormalize:
    def test_minmax_range(self, random_gray_f32):
        out = minmax_normalize(random_gray_f32)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_minmax_flat_image(self):
        flat = np.full((64, 64), 128.0, dtype=np.float32)
        out = minmax_normalize(flat)
        assert np.allclose(out, 0.0)

    def test_zscore_mean_zero(self, random_gray_f32):
        out = zscore_normalize(random_gray_f32)
        assert abs(float(out.mean())) < 0.01

    def test_to_float32_from_u8(self, random_gray_u8):
        out = to_float32(random_gray_u8)
        assert out.dtype == np.float32
        assert out.max() <= 1.0
        assert out.min() >= 0.0


class TestDenoise:
    @pytest.mark.parametrize("method", ["gaussian", "median"])
    def test_shape_preserved(self, method, random_gray_f32):
        out = denoise(random_gray_f32, method=method)
        assert out.shape == random_gray_f32.shape
        assert out.dtype == np.float32

    def test_unknown_method_raises(self, random_gray_f32):
        with pytest.raises(ValueError):
            denoise(random_gray_f32, method="bad_method")


class TestContrast:
    @pytest.mark.parametrize("method", ["clahe", "histogram_eq"])
    def test_shape_preserved(self, method, random_gray_f32):
        out = enhance_contrast(random_gray_f32, method=method)
        assert out.shape == random_gray_f32.shape

    def test_unknown_method_raises(self, random_gray_f32):
        with pytest.raises(ValueError):
            enhance_contrast(random_gray_f32, method="bad_method")


class TestResize:
    def test_output_size(self, random_gray_f32):
        out, scale, padding = letterbox(random_gray_f32, target_size=640)
        assert out.shape == (640, 640)
        assert out.dtype == np.float32

    def test_scale_positive(self, random_gray_f32):
        _, scale, _ = letterbox(random_gray_f32, target_size=640)
        assert scale > 0

    def test_square_image(self):
        square = np.random.rand(320, 320).astype(np.float32)
        out, scale, (pt, pl) = letterbox(square, target_size=640)
        assert out.shape == (640, 640)
        assert abs(scale - 2.0) < 1e-3

    def test_unletterbox_roundtrip(self):
        img = np.random.rand(200, 400).astype(np.float32)
        _, scale, (pt, pl) = letterbox(img, target_size=640)
        x1_pad, y1_pad, x2_pad, y2_pad = pl + 10, pt + 10, pl + 100, pt + 80
        x1, y1, x2, y2 = unletterbox_bbox(x1_pad, y1_pad, x2_pad, y2_pad, scale, pt, pl)
        assert x1 < x2 and y1 < y2


class TestSonarPreprocessor:
    def test_numpy_input(self, random_gray_u8, minimal_config):
        preprocessor = SonarPreprocessor(minimal_config)
        tensor, scale, padding = preprocessor(random_gray_u8)
        assert tensor.shape == (1, 640, 640)

    def test_rgb_input_converted(self, minimal_config):
        rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        preprocessor = SonarPreprocessor(minimal_config)
        tensor, scale, _ = preprocessor(rgb)
        assert tensor.shape == (1, 640, 640)

    def test_tensor_values_finite(self, random_gray_u8, minimal_config):
        preprocessor = SonarPreprocessor(minimal_config)
        tensor, _, _ = preprocessor(random_gray_u8)
        assert torch.isfinite(tensor).all()

    def test_custom_size(self, random_gray_u8):
        config = {
            "image_size": 416,
            "normalize": True,
            "denoise": {"enabled": False},
            "contrast": {"enabled": False},
        }
        preprocessor = SonarPreprocessor(config)
        tensor, _, _ = preprocessor(random_gray_u8)
        assert tensor.shape == (1, 416, 416)
