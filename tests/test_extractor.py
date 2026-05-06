"""Tests for memorebuilder.extractor."""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from memorebuilder.extractor import FeatureExtractor, _HIST_BINS, _SPATIAL_STRIPS


def _solid_image_buf(color=(128, 64, 200)):
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _gradient_image_buf():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)
    arr[:, :, 1] = 100
    arr[:, :, 2] = np.linspace(255, 0, 64, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Feature vector shape and range
    # ------------------------------------------------------------------

    def _save_tmp(self, buf) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(buf.read())
        return path

    def test_feature_vector_shape(self):
        path = self._save_tmp(_solid_image_buf())
        try:
            features = self.extractor.extract(path)
        finally:
            os.unlink(path)

        # 24 hist + 3 dominant + 96 spatial = 123
        expected_len = _HIST_BINS * 3 + 3 + 2 * _SPATIAL_STRIPS * 3 * 2
        assert features.shape == (expected_len,)

    def test_feature_values_in_unit_interval(self):
        path = self._save_tmp(_gradient_image_buf())
        try:
            features = self.extractor.extract(path)
        finally:
            os.unlink(path)

        assert features.min() >= 0.0 - 1e-9
        assert features.max() <= 1.0 + 1e-9

    def test_deterministic_extraction(self):
        """Extracting from the same image twice yields identical results."""
        path = self._save_tmp(_solid_image_buf(color=(10, 20, 30)))
        try:
            f1 = self.extractor.extract(path)
            f2 = self.extractor.extract(path)
        finally:
            os.unlink(path)

        np.testing.assert_array_equal(f1, f2)

    def test_different_images_give_different_features(self):
        p1 = self._save_tmp(_solid_image_buf(color=(255, 0, 0)))
        p2 = self._save_tmp(_solid_image_buf(color=(0, 255, 0)))
        try:
            f1 = self.extractor.extract(p1)
            f2 = self.extractor.extract(p2)
        finally:
            os.unlink(p1)
            os.unlink(p2)

        assert not np.allclose(f1, f2)

    def test_rgba_image_is_handled(self):
        """RGBA images should be accepted (converted to RGB internally)."""
        img = Image.new("RGBA", (32, 32), (200, 100, 50, 128))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path)
        try:
            features = self.extractor.extract(path)
        finally:
            os.unlink(path)

        assert features.ndim == 1
        assert features.size > 0

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            self.extractor.extract("/nonexistent/path/image.png")
