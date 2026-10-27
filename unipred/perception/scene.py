"""From an image to the feature vector of every object of a scene.

The objects of a scene are known by name and type (from the recorded
session, or the task). The detector finds the objects of each type, and the
detections are matched to the names: in the first image of a sequence from
left to right (by box centre) in the order of the names; afterwards by
appearance and place (Hungarian matching): an object's appearance is the
encoding of its crop with only its mask's pixels (the whole crop without a
mask), and the cost of a detection for an object is one minus the cosine of
their appearances plus a weight times the distance of their box centres
over the image diagonal. An object's appearance is the one it had where
first seen and the one it had last (the better match counts), as a toy
partly hidden in the gripper looks unlike itself on the table. A detection
whose appearance is less similar than min_similarity is not the object
(toys that look alike match by place). An object that is not detected
keeps its previous box and appearance (for example a toy hidden in the
gripper); boxes given explicitly (e.g. annotated for a hard object) replace
detection. The crop around each box is encoded into the object's features;
with box_features, the box itself follows (its centre and size over the
image's width and height), the object's pose in the image.

With a tracker (unipred/perception/tracker.py), the objects also keep their
names through the frames recorded between two images: each object of
track_types is followed from its mask in one image through the frames
after it, and in the next image the detection of its type whose mask
overlaps the followed mask most (an IoU of at least track_min_iou) is the
object; when none does, the followed mask itself is (a toy in the gripper
that the detector misses). The objects not in view to the tracker are
matched by appearance and place as above.

The settings come from a real-robot config (configs/real_robot.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, \
    Tuple, Union

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from unipred.perception.crops import crop_object
from unipred.perception.detector import Box, Detection, Detector
from unipred.perception.features import CropEncoder
from unipred.perception.tracker import Tracker


class PerceptionError(Exception):
    """An object of the scene (object_name, if one) could not be located."""

    def __init__(self, message: str, object_name: Optional[str] = None):
        super().__init__(message)
        self.object_name = object_name


@dataclass(frozen=True)
class SceneObservation:
    """The box, mask (None if neither detected nor followed in this image),
    features and appearance (for matching: unit vectors, of where it was
    first seen and of this image) of every object of a scene in one
    image."""
    boxes: Dict[str, Box]
    masks: Dict[str, Optional[np.ndarray]]
    features: Dict[str, np.ndarray]
    appearance: Dict[str, np.ndarray] = field(default_factory=dict)


class ScenePerceiver:
    """Locates and encodes the objects of a scene (see module doc)."""

    def __init__(self,
                 detector: Detector,
                 encoder: CropEncoder,
                 crop_padding: float = 0.15,
                 mask_crops: bool = False,
                 min_similarity: float = 0.7,
                 position_weight: float = 0.5,
                 tracker: Optional[Tracker] = None,
                 track_types: Optional[Sequence[str]] = None,
                 track_min_iou: float = 0.3,
                 box_features: bool = False) -> None:
        self.detector = detector
        self.encoder = encoder
        self.tracker = tracker
        self.box_features = box_features
        self._crop_padding = crop_padding
        self._mask_crops = mask_crops
        self._min_similarity = min_similarity
        self._position_weight = position_weight
        self._track_types = None if track_types is None else set(track_types)
        self._track_min_iou = track_min_iou
        # The observation whose masks the tracker follows the objects from.
        self._last: Optional[SceneObservation] = None

    @classmethod
    def from_config(cls,
                    path: str,
                    detector_backend: Optional[str] = None,
                    encoder_backend: Optional[str] = None,
                    tracker_backend: Optional[str] = None) -> ScenePerceiver:
        """A perceiver with the settings of a real-robot config; the
        backends can be overridden (e.g. with "fake" for tests; the
        tracker's with "none" for no tracker)."""
        config = load_robot_config(path)
        detector = Detector(detector_backend or config["detector"],
                            config["prompts"],
                            box_threshold=config["box_threshold"],
                            text_threshold=config["text_threshold"],
                            device=config["device"],
                            max_box_area=config["max_box_area"])
        encoder = CropEncoder(encoder_backend or config["encoder"],
                              device=config["device"])
        tracker = tracker_backend or config["tracker"]
        return cls(detector, encoder, config["crop_padding"],
                   config["mask_crops"], config["match_min_similarity"],
                   config["match_position_weight"],
                   None if tracker == "none" else Tracker(tracker),
                   config["track_types"], config["track_min_iou"],
                   config["box_features"])

    @property
    def feature_dim(self) -> int:
        """The length of an object's features."""
        return self.encoder.dim + (4 if self.box_features else 0)

    def follow(self, previous: Optional[SceneObservation],
               frames: Iterable[np.ndarray]) -> Optional[Dict[str,
                                                             np.ndarray]]:
        """The masks in the last frame of the objects that the tracker
        follows from previous (the observation perceived last) through the
        frames recorded after its image, oldest first, the last of them the
        image to perceive next (the objects in view there); None without a
        tracker, or if previous is not the observation perceived last."""
        if self.tracker is None or previous is None or previous is not \
                self._last:
            return None
        followed: Dict[str, np.ndarray] = {}
        for frame in frames:
            followed = self.tracker.follow(frame)
        return followed

    def perceive(
        self,
        image: np.ndarray,
        objects: Mapping[str, str],
        previous: Optional[Union[SceneObservation, Mapping[str, Box]]] = None,
        given: Optional[Mapping[str, Box]] = None,
        followed: Optional[Mapping[str, np.ndarray]] = None,
    ) -> SceneObservation:
        """The observation of the objects (name -> type) in an RGB image;
        previous is the observation of the image before (or its boxes
        only, for matching by place), followed the masks in this image of
        the objects that the tracker followed from it (follow(); None if
        it was not, and the tracks start afresh from this image)."""
        given = dict(given or {})
        tracks_go_on = followed is not None
        followed = dict(followed or {})
        if isinstance(previous, SceneObservation):
            previous_boxes: Dict[str, Box] = dict(previous.boxes)
            previous_looks = dict(previous.appearance)
        else:
            previous_boxes, previous_looks = dict(previous or {}), {}
        detections: Dict[str, List[Detection]] = {}
        for detection in self.detector.detect(image):
            detections.setdefault(detection.object_type, []).append(detection)
        diagonal = float(np.hypot(*image.shape[:2]))
        boxes: Dict[str, Box] = {}
        masks: Dict[str, Optional[np.ndarray]] = {}
        looks: Dict[str, np.ndarray] = {}
        for object_type in sorted(set(objects.values())):
            names = sorted(n for n, t in objects.items()
                           if t == object_type and n not in given)
            found = detections.get(object_type, [])
            found_looks = self._appearance(image, [(d.box, d.mask)
                                                   for d in found])
            matched = _match_followed([n for n in names if n in followed],
                                      found, followed, self._track_min_iou)
            rest = [j for j in range(len(found)) if j not in matched.values()]
            by_look = _match([n for n in names if n not in followed],
                             [found[j] for j in rest],
                             [found_looks[j] for j in rest], previous_boxes,
                             previous_looks, diagonal, self._min_similarity,
                             self._position_weight)
            matched.update((n, rest[j]) for n, j in by_look.items())
            for name in names:
                if name in matched:
                    index = matched[name]
                    boxes[name] = found[index].box
                    masks[name] = found[index].mask
                    first = np.atleast_2d(previous_looks.get(
                        name, found_looks[index]))[0]
                    looks[name] = np.stack([first, found_looks[index]])
                elif name in followed:
                    boxes[name] = _mask_box(followed[name])
                    masks[name] = followed[name]
                    if name in previous_looks:
                        looks[name] = previous_looks[name]
                elif name in previous_boxes:
                    boxes[name] = previous_boxes[name]
                    masks[name] = None
                    if name in previous_looks:
                        looks[name] = previous_looks[name]
                else:
                    raise PerceptionError(
                        f"{name} ({object_type}) was not detected and has no "
                        "earlier box; give its box explicitly.", name)
        for name, (x0, y0, x1, y1) in given.items():
            boxes[name] = (float(x0), float(y0), float(x1), float(y1))
            masks[name] = None
        missing = [n for n in objects if n not in looks]
        for name, look in zip(
                missing,
                self._appearance(image, [(boxes[n], None) for n in missing])):
            looks[name] = np.stack([look, look])
        names = sorted(objects)
        crops = [
            crop_object(image, boxes[n], self._crop_padding,
                        masks[n] if self._mask_crops else None) for n in names
        ]
        features = dict(zip(names, self.encoder.encode(crops)))
        if self.box_features:
            height, width = image.shape[:2]
            for name in names:
                x0, y0, x1, y1 = boxes[name]
                features[name] = np.concatenate([
                    features[name],
                    [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
                     (x1 - x0) / width, (y1 - y0) / height]
                ]).astype(np.float32)
        for name, vector in features.items():
            if not np.all(np.isfinite(vector)):
                raise PerceptionError(
                    f"The features of {name} (box {boxes[name]}) are not "
                    "finite.", name)
        observation = SceneObservation(boxes, masks, features, looks)
        if self.tracker is not None:
            # An object without a mask here (out of view) stays followed if
            # the tracks went on to this image.
            if not tracks_go_on:
                self.tracker.stop()
            self.tracker.start(
                image, {
                    n: m
                    for n, m in masks.items() if m is not None and (
                        self._track_types is None
                        or objects[n] in self._track_types)
                })
            self._last = observation
        return observation

    def _appearance(
            self, image: np.ndarray,
            regions: List[Tuple[Box, Optional[np.ndarray]]]) -> List[np.ndarray]:
        """Unit encodings of crops with only the mask's pixels (all of the
        crop without a mask)."""
        if not regions:
            return []
        encoded = self.encoder.encode([
            crop_object(image, box, self._crop_padding, mask)
            for box, mask in regions
        ])
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        return list(encoded / np.maximum(norms, 1e-12))


def _center(box: Box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _mask_box(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1),
            float(ys.max() + 1))


def _match_followed(names: List[str], detections: List[Detection],
                    followed: Mapping[str, np.ndarray],
                    min_iou: float) -> Dict[str, int]:
    """Assign detections (by index) of one type to the followed objects of
    that type, by the IoU of their masks (at least min_iou)."""
    if not names or not detections:
        return {}
    overlap = np.zeros((len(names), len(detections)))
    for i, name in enumerate(names):
        for j, detection in enumerate(detections):
            union = np.logical_or(followed[name], detection.mask).sum()
            overlap[i, j] = np.logical_and(followed[name],
                                           detection.mask).sum() / max(
                                               union, 1)
    rows, cols = linear_sum_assignment(-overlap)
    return {
        names[r]: int(c)
        for r, c in zip(rows, cols) if overlap[r, c] >= min_iou
    }


def _match(names: List[str], detections: List[Detection],
           looks: List[np.ndarray], previous: Mapping[str, Box],
           previous_looks: Mapping[str, np.ndarray], diagonal: float,
           min_similarity: float,
           position_weight: float) -> Dict[str, int]:
    """Assign detections (by index) of one type to the object names of that
    type: objects with a previous box by appearance and place (see module
    doc), the others to the remaining highest-scoring detections, from
    left to right."""
    tracked = [n for n in names if n in previous]
    matched: Dict[str, int] = {}
    if tracked and detections:
        cost = np.zeros((len(tracked), len(detections)))
        for i, name in enumerate(tracked):
            for j, detection in enumerate(detections):
                cost[i, j] = position_weight * float(
                    np.linalg.norm(
                        _center(previous[name]) -
                        _center(detection.box))) / diagonal
                if name in previous_looks:
                    similarity = float(
                        np.max(np.atleast_2d(previous_looks[name]) @ looks[j]))
                    cost[i, j] += 1.0 - similarity
                    if similarity < min_similarity:
                        cost[i, j] += 1e6
        rows, cols = linear_sum_assignment(cost)
        matched = {
            tracked[r]: int(c)
            for r, c in zip(rows, cols) if cost[r, c] < 1e6
        }
    untracked = [n for n in names if n not in previous]
    used = set(matched.values())
    rest = sorted((j for j in range(len(detections)) if j not in used),
                  key=lambda j: -detections[j].score)[:len(untracked)]
    rest.sort(key=lambda j: tuple(_center(detections[j].box)))
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
    "match_min_similarity": 0.7,
    "match_position_weight": 0.5,
    "tracker": "none",
    "track_types": None,
    "track_min_iou": 0.3,
    "box_features": False,
}


def load_robot_config(path: str) -> Dict:
    """A real-robot config (see configs/real_robot.yaml) with defaults."""
    with open(path, "r", encoding="utf-8") as f:
        config = {**_ROBOT_CONFIG_DEFAULTS, **(yaml.safe_load(f) or {})}
    if not config.get("prompts"):
        raise ValueError(f"{path} gives no detector prompts per object type.")
    return config
