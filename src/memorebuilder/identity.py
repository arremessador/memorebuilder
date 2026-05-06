"""Character identity preservation across generations.

A :class:`CharacterIdentity` bundles the polynomial coefficients that represent
a character's visual identity together with metadata, and provides helpers for:

* Serialising / deserialising identities to/from plain dictionaries (JSON-
  compatible).
* Computing the similarity between two identities so that parent–child or
  generation-to-generation continuity can be verified.

Similarity metric
-----------------
Two identities are compared via the normalised L2 distance between their
coefficient vectors after zero-padding the shorter one to match the longer::

    distance = ‖c1 - c2‖₂ / (‖c1‖₂ + ‖c2‖₂ + ε)

``similarity = 1 - distance``, clipped to [0, 1].  A value of 1.0 means the
identities are mathematically identical; a value ≥ 0.9 is considered a strong
match (same or closely related character).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_MATCH_THRESHOLD = 0.9


@dataclass
class CharacterIdentity:
    """Polynomial identity representation of a character.

    Attributes
    ----------
    coefficients:
        Polynomial coefficients (highest-degree first) encoding the
        character's visual features.
    label:
        Optional human-readable name or identifier for the character.
    metadata:
        Optional free-form dictionary for extra information (generation
        number, parent ID, source image path, …).
    """

    coefficients: np.ndarray
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        image_path: str,
        label: str = "",
        metadata: dict[str, Any] | None = None,
        *,
        degree: int = 12,
    ) -> "CharacterIdentity":
        """Create a :class:`CharacterIdentity` directly from an image file.

        Parameters
        ----------
        image_path:
            Path to the character image.
        label:
            Optional display name.
        metadata:
            Optional extra information to attach.
        degree:
            Polynomial degree used for encoding (default 12).

        Returns
        -------
        CharacterIdentity
        """
        from .extractor import FeatureExtractor
        from .encoder import EquationEncoder

        features = FeatureExtractor().extract(image_path)
        coefficients = EquationEncoder(degree=degree).encode(features)
        return cls(
            coefficients=coefficients,
            label=label,
            metadata=dict(metadata or {}, source_image=str(image_path)),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the identity to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Keys: ``"label"``, ``"coefficients"`` (list of floats),
            ``"metadata"``.
        """
        return {
            "label": self.label,
            "coefficients": self.coefficients.tolist(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        """Write the identity to a JSON file at *path*.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories must exist.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterIdentity":
        """Restore a :class:`CharacterIdentity` from a plain dictionary.

        Parameters
        ----------
        data:
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        CharacterIdentity

        Raises
        ------
        KeyError
            If required keys are missing.
        ValueError
            If ``"coefficients"`` is not a non-empty list of numbers.
        """
        if "coefficients" not in data:
            raise KeyError("'coefficients' key is required")
        coefficients = np.asarray(data["coefficients"], dtype=np.float64)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("'coefficients' must be a non-empty 1-D list of numbers")
        return cls(
            coefficients=coefficients,
            label=data.get("label", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterIdentity":
        """Load a :class:`CharacterIdentity` from a JSON file.

        Parameters
        ----------
        path:
            Path to the JSON file written by :meth:`save`.

        Returns
        -------
        CharacterIdentity
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Comparison / matching
    # ------------------------------------------------------------------

    def similarity(self, other: "CharacterIdentity") -> float:
        """Compute normalised similarity in [0, 1] against *other*.

        Coefficients are zero-padded to the same length before comparison.

        Parameters
        ----------
        other:
            Another :class:`CharacterIdentity`.

        Returns
        -------
        float
            1.0 for identical identities, lower values for more different ones.
        """
        a = self.coefficients
        b = other.coefficients
        # Zero-pad to equal length
        max_len = max(len(a), len(b))
        a = np.pad(a, (max_len - len(a), 0))
        b = np.pad(b, (max_len - len(b), 0))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        distance = np.linalg.norm(a - b) / (norm_a + norm_b + 1e-12)
        return float(np.clip(1.0 - distance, 0.0, 1.0))

    def matches(self, other: "CharacterIdentity", threshold: float = _MATCH_THRESHOLD) -> bool:
        """Return True if *other* is considered the same character.

        Parameters
        ----------
        other:
            Another :class:`CharacterIdentity`.
        threshold:
            Minimum similarity score to consider a match (default 0.9).

        Returns
        -------
        bool
        """
        return self.similarity(other) >= threshold

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CharacterIdentity(label={self.label!r}, "
            f"degree={len(self.coefficients) - 1})"
        )
