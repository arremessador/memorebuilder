"""Feature extraction from character images.

Extracts a fixed-length numerical feature vector from an image that captures
the visual identity of a character:

* Normalised RGB colour histograms (8 bins each channel → 24 values)
* Dominant-colour centroid in RGB space (3 values)
* Horizontal and vertical colour-moment vectors (mean + std per channel,
  split into 8 equal strips → 2 × 8 × 3 × 2 = 96 values)

Total raw features: 24 + 3 + 96 = 123, normalised to [0, 1] before encoding.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


_HIST_BINS = 8
_SPATIAL_STRIPS = 8


def _colour_histograms(arr: np.ndarray) -> np.ndarray:
    """Return a normalised colour histogram for each RGB channel.

    Parameters
    ----------
    arr:
        H × W × 3 uint8 array.

    Returns
    -------
    np.ndarray
        1-D array of length ``_HIST_BINS * 3`` with values in [0, 1].
    """
    features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=_HIST_BINS, range=(0, 256))
        total = hist.sum()
        features.append(hist / total if total > 0 else hist.astype(float))
    return np.concatenate(features)


def _dominant_colour(arr: np.ndarray) -> np.ndarray:
    """Return the mean (dominant) colour of the image normalised to [0, 1].

    Parameters
    ----------
    arr:
        H × W × 3 uint8 array.

    Returns
    -------
    np.ndarray
        1-D array of length 3.
    """
    return arr.mean(axis=(0, 1)) / 255.0


def _spatial_colour_moments(arr: np.ndarray) -> np.ndarray:
    """Return mean and std of each channel across horizontal/vertical strips.

    The image is divided into ``_SPATIAL_STRIPS`` equal strips along each axis.
    For every strip we compute the per-channel mean and standard deviation,
    yielding a feature vector of length
    ``2 * _SPATIAL_STRIPS * 3_channels * 2_moments = 96``.

    Parameters
    ----------
    arr:
        H × W × 3 uint8 array.

    Returns
    -------
    np.ndarray
        1-D array of length ``2 * _SPATIAL_STRIPS * 3 * 2``, normalised to
        [0, 1] (values divided by 255).
    """
    h, w = arr.shape[:2]
    features = []
    for axis, size in ((0, h), (1, w)):
        strip_size = max(size // _SPATIAL_STRIPS, 1)
        for i in range(_SPATIAL_STRIPS):
            start = i * strip_size
            end = start + strip_size if i < _SPATIAL_STRIPS - 1 else size
            strip = arr[start:end, :, :] if axis == 0 else arr[:, start:end, :]
            features.append(strip.mean(axis=(0, 1)) / 255.0)
            features.append(strip.std(axis=(0, 1)) / 255.0)
    return np.concatenate(features)


class FeatureExtractor:
    """Extract a deterministic feature vector from a character image.

    Usage
    -----
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract("character.png")
    >>> features.shape
    (123,)
    """

    def extract(self, image_path: str) -> np.ndarray:
        """Extract features from *image_path* and return a 1-D float64 array.

        The image is converted to RGB mode before processing, so RGBA, palette,
        and greyscale images are handled transparently.

        Parameters
        ----------
        image_path:
            Path to an image file (any format supported by Pillow).

        Returns
        -------
        np.ndarray
            Feature vector of shape ``(123,)`` with values in [0, 1].

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        OSError
            If the file cannot be opened as an image.
        """
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)

        hist_features = _colour_histograms(arr)
        dominant_colour = _dominant_colour(arr)
        spatial_features = _spatial_colour_moments(arr)

        return np.concatenate([hist_features, dominant_colour, spatial_features])
