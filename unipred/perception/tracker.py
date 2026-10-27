"""Following objects from frame to frame, so that they keep their names
between the keyframes of a recording (see unipred/perception/scene.py).

A tracker starts from each object's mask in one image and follows the
objects through the frames after it, one at a time. An object out of view
in a frame (behind the arm, say, or the tracker unsure) has no mask there,
but is still followed, to be found again when it comes back into view.

Backends:

- "sam3": SAM3's video tracker on a SAM3 server (CDG's perception servers;
  unipred/perception/sam3.py), a track per object;
- "fake": for tests and the synthetic fixture (unipred/perception/
  detector.py draws the objects of a type in one colour): an object is the
  region of its colour that overlaps most where it was last seen.
"""

from __future__ import annotations

import uuid
from typing import Dict, Mapping

import numpy as np
from scipy import ndimage

from unipred.perception import sam3
from unipred.perception.detector import FAKE_MIN_AREA


class Tracker:
    """Follows the masks (H x W booleans) of named objects (see module
    doc)."""

    def __init__(self, backend: str) -> None:
        if backend not in ("sam3", "fake"):
            raise ValueError(f"Unknown tracker backend: {backend!r} (use "
                             "sam3 or fake).")
        self.backend = backend
        self._prefix = f"unipred-{uuid.uuid4().hex[:8]}-"
        # The mask of each followed object where it was last seen.
        self._masks: Dict[str, np.ndarray] = {}
        self._colors: Dict[str, np.ndarray] = {}

    @property
    def followed(self) -> frozenset:
        """The names of the objects followed."""
        return frozenset(self._masks)

    def start(self, image: np.ndarray, masks: Mapping[str,
                                                      np.ndarray]) -> None:
        """Follow these objects from their masks in an RGB image (afresh,
        if followed already); the other objects followed stay so."""
        for name, mask in masks.items():
            if not mask.any():
                continue
            if self.backend == "sam3":
                sam3.track_start(image, mask, self._prefix + name)
            else:
                self._colors[name] = np.median(image[mask], axis=0)
            self._masks[name] = mask

    def follow(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """The masks of the followed objects in view in the next RGB
        image."""
        found = {}
        for name in sorted(self._masks):
            if self.backend == "sam3":
                _, score, mask = sam3.track(image, self._prefix + name)
                if score < 0.5:
                    continue
            else:
                mask = self._follow_fake(image, name)
            if mask.any():
                found[name] = self._masks[name] = mask
        return found

    def stop(self) -> None:
        """Follow no object."""
        if self.backend == "sam3":
            for name in self._masks:
                sam3.track_reset(self._prefix + name)
        self._masks.clear()
        self._colors.clear()

    def _follow_fake(self, image: np.ndarray, name: str) -> np.ndarray:
        close = np.abs(image.astype(np.int16) -
                       self._colors[name]).max(axis=-1) <= 8
        labels, _ = ndimage.label(close)
        overlap = np.bincount(labels[self._masks[name]].ravel())
        overlap[0] = 0
        region = int(np.argmax(overlap))
        if overlap[region] == 0:
            return np.zeros(image.shape[:2], bool)
        mask = labels == region
        return mask if mask.sum() >= FAKE_MIN_AREA else np.zeros_like(mask)
