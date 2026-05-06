"""Encode a feature vector as a polynomial equation.

The feature vector produced by :class:`~memorebuilder.extractor.FeatureExtractor`
is mapped onto a set of evaluation points and a polynomial is fitted through
those points.  The resulting coefficients *are* the mathematical representation
of the character's identity.

Mathematical model
------------------
Given a feature vector **f** of length N:

1. Evaluation points: ``x_i = i / (N - 1)`` for i ∈ {0, …, N-1}  (uniform
   grid over [0, 1]).
2. A polynomial P of degree ``degree`` is least-squares fitted so that
   ``P(x_i) ≈ f_i``.
3. The coefficients ``[a_0, a_1, …, a_degree]`` are returned.

The polynomial can be reconstructed and evaluated at any future generation to
verify or compare identities.
"""

from __future__ import annotations

import numpy as np


_DEFAULT_DEGREE = 12


class EquationEncoder:
    """Encode a feature vector as a polynomial identity equation.

    Parameters
    ----------
    degree:
        Degree of the fitted polynomial.  Higher values capture finer
        feature differences at the cost of sensitivity to noise.
        Defaults to 12 which provides a good balance for the 123-dimensional
        feature vectors produced by :class:`~memorebuilder.extractor.FeatureExtractor`.

    Usage
    -----
    >>> encoder = EquationEncoder()
    >>> coeffs = encoder.encode(features)
    >>> equation_str = encoder.to_equation_string(coeffs)
    """

    def __init__(self, degree: int = _DEFAULT_DEGREE) -> None:
        if degree < 1:
            raise ValueError("degree must be a positive integer")
        self.degree = degree

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Fit a polynomial to *features* and return its coefficients.

        Parameters
        ----------
        features:
            1-D float array of length ≥ 2.

        Returns
        -------
        np.ndarray
            Coefficient array of shape ``(degree + 1,)`` ordered from the
            highest degree term to the constant term (same convention as
            :func:`numpy.polyfit`).

        Raises
        ------
        ValueError
            If *features* is not a 1-D array or has fewer than 2 elements.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 1 or features.size < 2:
            raise ValueError("features must be a 1-D array with at least 2 elements")

        n = features.size
        x = np.linspace(0.0, 1.0, n)
        coeffs = np.polyfit(x, features, min(self.degree, n - 1))
        return coeffs

    def evaluate(self, coeffs: np.ndarray, x: float | np.ndarray) -> np.ndarray:
        """Evaluate the polynomial with *coeffs* at point(s) *x*.

        Parameters
        ----------
        coeffs:
            Coefficient array as returned by :meth:`encode`.
        x:
            Scalar or array of evaluation points.

        Returns
        -------
        np.ndarray
            Polynomial values at *x*.
        """
        return np.polyval(coeffs, x)

    def to_equation_string(self, coeffs: np.ndarray) -> str:
        """Return a human-readable polynomial equation string.

        The polynomial is written in standard form::

            P(x) = a_n·xⁿ + … + a_1·x + a_0

        Parameters
        ----------
        coeffs:
            Coefficient array (highest-degree first) as returned by
            :meth:`encode`.

        Returns
        -------
        str
            Equation string, e.g. ``"P(x) = 3.14·x² + -1.00·x + 0.50"``.
        """
        degree = len(coeffs) - 1
        terms = []
        superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

        for i, coef in enumerate(coeffs):
            power = degree - i
            coef_str = f"{coef:.6g}"
            if power == 0:
                terms.append(coef_str)
            elif power == 1:
                terms.append(f"{coef_str}·x")
            else:
                exp = str(power).translate(superscripts)
                terms.append(f"{coef_str}·x{exp}")

        return "P(x) = " + " + ".join(terms)
