"""Tests for memorebuilder.cli."""

from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from memorebuilder.cli import main


def _solid_image_buf(color=(128, 64, 200)):
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _tmp_image(buf) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(buf.read())
    return path


class TestCLIEncode:
    def test_encode_prints_equation(self, capsys):
        path = _tmp_image(_solid_image_buf())
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["encode", path])
        finally:
            os.unlink(path)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "P(x) =" in captured.out

    def test_encode_saves_json(self, tmp_path):
        path = _tmp_image(_solid_image_buf())
        out_path = str(tmp_path / "identity.json")
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["encode", path, "--label", "TestHero", "--output", out_path])
        finally:
            os.unlink(path)
        assert exc_info.value.code == 0
        with open(out_path) as fh:
            data = json.load(fh)
        assert data["label"] == "TestHero"
        assert "coefficients" in data

    def test_encode_nonexistent_image_exits_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["encode", "/no/such/file.png"])
        assert exc_info.value.code == 1

    def test_encode_custom_degree(self, capsys):
        path = _tmp_image(_solid_image_buf())
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["encode", path, "--degree", "5"])
        finally:
            os.unlink(path)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # degree 5 → 6 coefficients
        assert "Coefficients (6)" in captured.out


class TestCLICompare:
    def test_compare_same_image_exits_0(self, capsys):
        path = _tmp_image(_solid_image_buf(color=(100, 150, 200)))
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["compare", path, path])
        finally:
            os.unlink(path)
        assert exc_info.value.code == 0

    def test_compare_different_images_exits_2(self, capsys):
        p1 = _tmp_image(_solid_image_buf(color=(255, 0, 0)))
        p2 = _tmp_image(_solid_image_buf(color=(0, 0, 255)))
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["compare", p1, p2])
        finally:
            os.unlink(p1)
            os.unlink(p2)
        # Different characters → no match → exit code 2
        assert exc_info.value.code == 2

    def test_compare_prints_similarity_score(self, capsys):
        path = _tmp_image(_solid_image_buf())
        try:
            with pytest.raises(SystemExit):
                main(["compare", path, path])
        finally:
            os.unlink(path)
        captured = capsys.readouterr()
        assert "Similarity score" in captured.out

    def test_compare_json_with_image(self, tmp_path, capsys):
        """Comparing a saved identity JSON against the same image should match."""
        path = _tmp_image(_solid_image_buf(color=(80, 160, 240)))
        out_json = str(tmp_path / "id.json")
        try:
            with pytest.raises(SystemExit):
                main(["encode", path, "--output", out_json])
            with pytest.raises(SystemExit) as exc_info:
                main(["compare", out_json, path])
        finally:
            os.unlink(path)
        assert exc_info.value.code == 0

    def test_compare_missing_file_exits_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "/no/such.png", "/also/nope.png"])
        assert exc_info.value.code == 1
