"""Shared test fixtures for memorebuilder tests."""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image


def _make_solid_image_buf(
    width: int = 64,
    height: int = 64,
    color: tuple[int, int, int] = (128, 64, 200),
) -> io.BytesIO:
    """Return an in-memory PNG buffer of a solid-colour image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _make_gradient_image_buf(
    width: int = 64,
    height: int = 64,
) -> io.BytesIO:
    """Return an in-memory PNG buffer with a simple horizontal gradient."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    arr[:, :, 1] = 100
    arr[:, :, 2] = np.linspace(255, 0, width, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture()
def solid_image_buf():
    """In-memory PNG buffer of a solid-colour image."""
    return _make_solid_image_buf()


@pytest.fixture()
def gradient_image_buf():
    """In-memory PNG buffer of a gradient image."""
    return _make_gradient_image_buf()
