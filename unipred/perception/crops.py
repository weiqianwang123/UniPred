"""Object crops: the image region around a detected object, which the crop
encoder (unipred/perception/features.py) turns into the object's
features."""

from __future__ import annotations

from typing import Optional

import numpy as np

from unipred.perception.detector import Box


def crop_object(image: np.ndarray,
                box: Box,
                padding: float = 0.15,
                mask: Optional[np.ndarray] = None,
                background: int = 0) -> np.ndarray:
    """The square crop around a box, enlarged on every side by padding
    times the box's larger side and clipped to the image.

    The crop keeps the object's surroundings, which predicates about its
    relations to other objects depend on. With a mask (H x W booleans over
    the image), the pixels outside it are set to background.
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * padding)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(np.floor(cx - side / 2), 0, width - 1))
    top = int(np.clip(np.floor(cy - side / 2), 0, height - 1))
    right = int(np.clip(np.ceil(cx + side / 2), left + 1, width))
    bottom = int(np.clip(np.ceil(cy + side / 2), top + 1, height))
    crop = image[top:bottom, left:right].copy()
    if mask is not None:
        crop[~mask[top:bottom, left:right]] = background
    return crop
