"""Tests for memorebuilder.encoder."""

from __future__ import annotations

import numpy as np
import pytest

from memorebuilder.encoder import EquationEncoder, _DEFAULT_DEGREE


class TestEquationEncoder:
    def setup_method(self):
        self.encoder = EquationEncoder()

    # ------------------------------------------------------------------
    # encode()
    # ------------------------------------------------------------------

    def test_coefficients_shape(self):
        features = np.linspace(0, 1, 123)
        coeffs = self.encoder.encode(features)
        assert coeffs.shape == (_DEFAULT_DEGREE + 1,)

    def test_encode_short_feature_vector(self):
        """Degree is capped at n-1 for very short inputs."""
        features = np.array([0.1, 0.5, 0.9])
        encoder = EquationEncoder(degree=10)
        coeffs = encoder.encode(features)
        # degree is min(10, 3-1) = 2 → 3 coefficients
        assert coeffs.shape == (3,)

    def test_encode_is_deterministic(self):
        features = np.random.default_rng(42).random(50)
        c1 = self.encoder.encode(features)
        c2 = self.encoder.encode(features)
        np.testing.assert_array_equal(c1, c2)

    def test_encode_constant_features_gives_approx_constant_polynomial(self):
        """A constant feature vector should yield a near-constant polynomial."""
        features = np.full(50, 0.5)
        coeffs = self.encoder.encode(features)
        # High-degree coefficients should be near zero
        assert abs(self.encoder.evaluate(coeffs, 0.5) - 0.5) < 0.01

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError):
            EquationEncoder(degree=0)

    def test_invalid_features_not_1d_raises(self):
        with pytest.raises(ValueError):
            self.encoder.encode(np.zeros((3, 3)))

    def test_invalid_features_too_short_raises(self):
        with pytest.raises(ValueError):
            self.encoder.encode(np.array([0.5]))

    # ------------------------------------------------------------------
    # evaluate()
    # ------------------------------------------------------------------

    def test_evaluate_scalar(self):
        features = np.linspace(0, 1, 20)
        coeffs = self.encoder.encode(features)
        val = self.encoder.evaluate(coeffs, 0.5)
        assert np.isscalar(val) or val.shape == ()

    def test_evaluate_array(self):
        features = np.linspace(0, 1, 20)
        coeffs = self.encoder.encode(features)
        x = np.linspace(0, 1, 10)
        vals = self.encoder.evaluate(coeffs, x)
        assert vals.shape == (10,)

    # ------------------------------------------------------------------
    # to_equation_string()
    # ------------------------------------------------------------------

    def test_equation_string_starts_with_px(self):
        features = np.linspace(0, 1, 10)
        coeffs = self.encoder.encode(features)
        eq = self.encoder.to_equation_string(coeffs)
        assert eq.startswith("P(x) =")

    def test_equation_string_has_correct_number_of_terms(self):
        encoder = EquationEncoder(degree=3)
        features = np.linspace(0, 1, 20)
        coeffs = encoder.encode(features)
        eq = self.encoder.to_equation_string(coeffs)
        # Should contain as many " + " separators as len(coeffs) - 1
        assert eq.count(" + ") == len(coeffs) - 1

    def test_equation_string_contains_superscript_for_high_degree(self):
        encoder = EquationEncoder(degree=3)
        features = np.linspace(0, 1, 20)
        coeffs = encoder.encode(features)
        eq = encoder.to_equation_string(coeffs)
        # degree-3 term should contain a superscript ³
        assert "³" in eq
