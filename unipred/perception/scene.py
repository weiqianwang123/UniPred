"""From an image to the feature vector of every object of a scene.

The objects of a scene are known by name and type (from the recorded
session, or the task). The detector finds the objects of each type, and the
detections are matched to the names: in the first image of a sequence from
left to right (by box centre) in the order of the names, then each object to
the nearest detection of its type, by its box in the previous image. An
object that is not detected keeps its previous box (for example a toy
hidden in the gripper); boxes given explicitly (e.g. annotated for a hard
object) replace detection. The crop around each box is encoded into the
object's features.

The settings come from a real-robot config (configs/real_robot.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from unipred.perception.crops import crop_object
from unipred.perception.detector import Box, Detection, Detector
from unipred.perception.features import CropEncoder


class PerceptionError(Exception):
    """An object of the scene could not be located."""


@dataclass(frozen=True)
class SceneObservation:
    """The box, mask (None if not detected in this image) and features of
    every object of a scene in one image."""
    boxes: Dict[str, Box]
    masks: Dict[str, Optional[np.ndarray]]
    features: Dict[str, np.ndarray]


class ScenePerceiver:
    """Locates and encodes the objects of a scene (see module doc)."""

    def __init__(self,
                 detector: Detector,
                 encoder: CropEncoder,
                 crop_padding: float = 0.15,
                 mask_crops: bool = False) -> None:
        self.detector = detector
        self.encoder = encoder
        self._crop_padding = crop_padding
        self._mask_crops = mask_crops

    @classmethod
    def from_config(cls,
                    path: str,
                    detector_backend: Optional[str] = None,
                    encoder_backend: Optional[str] = None) -> ScenePerceiver:
        """A perceiver with the settings of a real-robot config; the
        backends can be overridden (e.g. with "fake" for tests)."""
        config = load_robot_config(path)
        detector = Detector(detector_backend or config["detector"],
                            config["prompts"],
                            box_threshold=config["box_threshold"],
                            text_threshold=config["text_threshold"],
                            device=config["device"],
                            max_box_area=config["max_box_area"])
        encoder = CropEncoder(encoder_backend or config["encoder"],
                              device=config["device"])
        return cls(detector, encoder, config["crop_padding"],
                   config["mask_crops"])

    def perceive(
        self,
        image: np.ndarray,
        objects: Mapping[str, str],
        previous: Optional[Mapping[str, Box]] = None,
        given: Optional[Mapping[str, Box]] = None,
    ) -> SceneObservation:
        """The observation of the objects (name -> type) in an RGB image."""
        given = dict(given or {})
        detections: Dict[str, List[Detection]] = {}
        for detection in self.detector.detect(image):
            detections.setdefault(detection.object_type, []).append(detection)
        boxes: Dict[str, Box] = {}
        masks: Dict[str, Optional[np.ndarray]] = {}
        for object_type in sorted(set(objects.values())):
            names = sorted(n for n, t in objects.items()
                           if t == object_type and n not in given)
            matched = _match(names, detections.get(object_type, []),
                             previous)
            for name in names:
                if name in matched:
                    boxes[name] = matched[name].box
                    masks[name] = matched[name].mask
                elif previous is not None and name in previous:
                    boxes[name] = previous[name]
                    masks[name] = None
                else:
                    raise PerceptionError(
                        f"{name} ({object_type}) was not detected and has no "
                        "earlier box; give its box explicitly.")
        for name, (x0, y0, x1, y1) in given.items():
            boxes[name] = (float(x0), float(y0), float(x1), float(y1))
            masks[name] = None
        names = sorted(objects)
        crops = [
            crop_object(image, boxes[n], self._crop_padding,
                        masks[n] if self._mask_crops else None) for n in names
        ]
        features = dict(zip(names, self.encoder.encode(crops)))
        for name, vector in features.items():
            if not np.all(np.isfinite(vector)):
                raise PerceptionError(
                    f"The features of {name} (box {boxes[name]}) are not "
                    "finite.")
        return SceneObservation(boxes, masks, features)


def _center(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _match(names: List[str], detections: List[Detection],
           previous: Optional[Mapping[str, Box]]) -> Dict[str, Detection]:
    """Assign detections of one type to the object names of that type:
    objects with a previous box to the nearest detections, the others to
    the remaining highest-scoring detections, from left to right."""
    tracked = [n for n in names if previous is not None and n in previous]
    matched: Dict[str, Detection] = {}
    if tracked and detections:
        assert previous is not None
        cost = np.array([[
            float(np.linalg.norm(_center(previous[n]) - _center(d.box)))
            for d in detections
        ] for n in tracked])
        rows, cols = linear_sum_assignment(cost)
        matched = {tracked[r]: detections[c] for r, c in zip(rows, cols)}
    untracked = [n for n in names if n not in tracked]
    used = {id(d) for d in matched.values()}
    rest = sorted((d for d in detections if id(d) not in used),
                  key=lambda d: -d.score)[:len(untracked)]
    rest.sort(key=lambda d: tuple(_center(d.box)))
    matched.update(zip(untracked, rest))
    return matched


_ROBOT_CONFIG_DEFAULTS = {
    "detector": "groundingdino_sam2",
    "encoder": "dinov3",
    "device": "cuda",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
    "crop_padding": 0.15,
    "mask_crops": False,
    "cameras": [0],
    "max_box_area": {},
}


def load_robot_config(path: str) -> Dict:
    """A real-robot config (see configs/real_robot.yaml) with defaults."""
    with open(path, "r", encoding="utf-8") as f:
        config = {**_ROBOT_CONFIG_DEFAULTS, **(yaml.safe_load(f) or {})}
    if not config.get("prompts"):
        raise ValueError(f"{path} gives no detector prompts per object type.")
    return config
