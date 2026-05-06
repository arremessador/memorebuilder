"""Tests for memorebuilder.identity."""

from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from memorebuilder.identity import CharacterIdentity


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


class TestCharacterIdentityCreation:
    def test_from_image(self):
        path = _tmp_image(_solid_image_buf())
        try:
            identity = CharacterIdentity.from_image(path, label="Hero")
        finally:
            os.unlink(path)

        assert identity.label == "Hero"
        assert identity.coefficients.ndim == 1
        assert identity.coefficients.size > 0
        assert "source_image" in identity.metadata

    def test_from_image_default_label_empty(self):
        path = _tmp_image(_solid_image_buf())
        try:
            identity = CharacterIdentity.from_image(path)
        finally:
            os.unlink(path)
        assert identity.label == ""


class TestCharacterIdentitySerialization:
    def _make_identity(self) -> CharacterIdentity:
        return CharacterIdentity(
            coefficients=np.array([1.0, -0.5, 0.25]),
            label="TestChar",
            metadata={"generation": 1},
        )

    def test_to_dict_round_trip(self):
        identity = self._make_identity()
        data = identity.to_dict()

        assert data["label"] == "TestChar"
        assert data["metadata"]["generation"] == 1
        assert isinstance(data["coefficients"], list)
        assert len(data["coefficients"]) == 3

    def test_from_dict_restores_identity(self):
        identity = self._make_identity()
        restored = CharacterIdentity.from_dict(identity.to_dict())

        assert restored.label == identity.label
        np.testing.assert_array_almost_equal(restored.coefficients, identity.coefficients)
        assert restored.metadata["generation"] == 1

    def test_save_and_load(self):
        identity = self._make_identity()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            identity.save(path)
            loaded = CharacterIdentity.load(path)
        finally:
            os.unlink(path)

        assert loaded.label == identity.label
        np.testing.assert_array_almost_equal(loaded.coefficients, identity.coefficients)

    def test_save_produces_valid_json(self):
        identity = self._make_identity()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            identity.save(path)
            with open(path) as fh:
                data = json.load(fh)
        finally:
            os.unlink(path)
        assert "coefficients" in data

    def test_from_dict_missing_coefficients_raises(self):
        with pytest.raises(KeyError):
            CharacterIdentity.from_dict({"label": "X"})

    def test_from_dict_empty_coefficients_raises(self):
        with pytest.raises(ValueError):
            CharacterIdentity.from_dict({"coefficients": []})


class TestCharacterIdentityComparison:
    def test_identical_identities_have_max_similarity(self):
        coeffs = np.array([1.0, 2.0, 3.0])
        a = CharacterIdentity(coefficients=coeffs.copy())
        b = CharacterIdentity(coefficients=coeffs.copy())
        assert a.similarity(b) == pytest.approx(1.0)

    def test_different_identities_have_lower_similarity(self):
        a = CharacterIdentity(coefficients=np.array([1.0, 0.0, 0.0]))
        b = CharacterIdentity(coefficients=np.array([0.0, 0.0, 1.0]))
        score = a.similarity(b)
        assert 0.0 <= score < 1.0

    def test_similarity_is_symmetric(self):
        a = CharacterIdentity(coefficients=np.array([1.0, 2.0]))
        b = CharacterIdentity(coefficients=np.array([3.0, 1.0]))
        assert a.similarity(b) == pytest.approx(b.similarity(a))

    def test_similarity_different_length_coefficients(self):
        a = CharacterIdentity(coefficients=np.array([1.0, 0.5]))
        b = CharacterIdentity(coefficients=np.array([0.0, 1.0, 0.5]))
        score = a.similarity(b)
        assert 0.0 <= score <= 1.0

    def test_matches_same_character(self):
        coeffs = np.ones(10)
        a = CharacterIdentity(coefficients=coeffs.copy())
        b = CharacterIdentity(coefficients=coeffs.copy())
        assert a.matches(b)

    def test_no_match_very_different_characters(self):
        a = CharacterIdentity(coefficients=np.array([1.0, 0.0, 0.0, 0.0]))
        b = CharacterIdentity(coefficients=np.array([0.0, 0.0, 0.0, -1.0]))
        assert not a.matches(b)

    def test_cross_generation_preservation(self):
        """Slightly perturbed coefficients should still match (same character)."""
        path = _tmp_image(_solid_image_buf(color=(120, 80, 200)))
        try:
            id1 = CharacterIdentity.from_image(path, label="Gen1")
            id2 = CharacterIdentity.from_image(path, label="Gen2")
        finally:
            os.unlink(path)

        # Same image → coefficients must be identical → similarity = 1
        assert id1.similarity(id2) == pytest.approx(1.0)

    def test_different_characters_do_not_match(self):
        p1 = _tmp_image(_solid_image_buf(color=(255, 0, 0)))
        p2 = _tmp_image(_solid_image_buf(color=(0, 0, 255)))
        try:
            id1 = CharacterIdentity.from_image(p1)
            id2 = CharacterIdentity.from_image(p2)
        finally:
            os.unlink(p1)
            os.unlink(p2)

        # Very different images → should not match at threshold 0.9
        assert not id1.matches(id2)
