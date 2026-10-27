"""Object detection: boxes and masks for the objects of each type from a
text prompt per type.

Backends:

- "groundingdino_sam2": GroundingDINO boxes refined into SAM2 masks,
  through transformers, from their installed weights
  (unipred/perception/weights.py);
- "sam3": SAM3's text-prompted instance segmentation, from a SAM3 server
  (CDG's perception servers; unipred/perception/sam3.py);
- "fake": no network weights, for tests and the synthetic fixture of
  examples/real_robot. The objects of each type are drawn as regions of the
  colour fake_colors() gives the type; the detector returns every connected
  region of that colour of at least FAKE_MIN_AREA pixels (smaller ones are
  slivers of objects covered by others).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, \
    Union

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from unipred.perception import sam3, weights

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
FAKE_MIN_AREA = 9


@dataclass(frozen=True)
class Detection:
    """A detected object: its type, box, score and mask (H x W booleans)."""
    object_type: str
    box: Box
    score: float
    mask: np.ndarray


# Well separated colours, none of them grey.
_FAKE_PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
                 (245, 130, 48), (145, 30, 180), (70, 240, 240),
                 (240, 50, 230), (210, 245, 60), (0, 0, 128), (170, 110, 40),
                 (0, 128, 128)]


def fake_colors(object_types: Iterable[str]) -> Dict[str, Tuple[int, int, int]]:
    """The colour of each object type for the fake backend: a palette in
    the order of the type names."""
    types = sorted(set(object_types))
    if len(types) > len(_FAKE_PALETTE):
        raise ValueError(f"The fake detector supports at most "
                         f"{len(_FAKE_PALETTE)} object types.")
    return dict(zip(types, _FAKE_PALETTE))


class Detector:
    """Detects the objects of the types in prompts (type -> a text prompt,
    or a list of them for a type whose objects look different, e.g. toys:
    [red cube, yellow ball]). A detection of a type in max_box_area larger
    than that fraction of the image is dropped (e.g. a box around the robot
    arm that GroundingDINO takes for a toy)."""

    def __init__(self,
                 backend: str,
                 prompts: Mapping[str, Union[str, Sequence[str]]],
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 device: str = "cpu",
                 max_box_area: Optional[Mapping[str, float]] = None) -> None:
        if backend not in ("groundingdino_sam2", "sam3", "fake"):
            raise ValueError(f"Unknown detector backend: {backend!r} (use "
                             "groundingdino_sam2, sam3 or fake).")
        self.backend = backend
        self._prompts = {
            t: [p] if isinstance(p, str) else list(p)
            for t, p in prompts.items()
        }
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._max_box_area = dict(max_box_area or {})
        unknown = set(self._max_box_area) - set(self._prompts)
        if unknown:
            raise ValueError(f"max_box_area names types without prompts: "
                             f"{sorted(unknown)}.")
        self._device = torch.device(device if torch.cuda.is_available()
                                    or device == "cpu" else "cpu")
        if backend == "groundingdino_sam2":
            self._load_models()

    def detect(self, image: np.ndarray) -> List[Detection]:
        """The detections in an RGB image (H x W x 3, uint8)."""
        if self.backend == "fake":
            return self._detect_fake(image)
        if self.backend == "sam3":
            return self._detect_sam3(image)
        return self._detect_models(image)

    def _detect_sam3(self, image: np.ndarray) -> List[Detection]:
        """Every prompt of every type sent to SAM3; its scores are compared
        with box_threshold."""
        found, masks = [], {}
        area = image.shape[0] * image.shape[1]
        for object_type, prompts in self._prompts.items():
            for prompt in prompts:
                for box, score, mask in sam3.segment_text(image, prompt):
                    if score >= self._box_threshold and self._small_enough(
                            object_type, box, area):
                        found.append((object_type, box, score))
                        masks[(object_type, box, score)] = mask
        return [
            Detection(object_type, box, score, masks[(object_type, box,
                                                      score)])
            for object_type, box, score in _suppress_duplicates(found)
        ]

    def _small_enough(self, object_type: str, box: Box,
                      image_area: float) -> bool:
        limit = self._max_box_area.get(object_type)
        area = (box[2] - box[0]) * (box[3] - box[1])
        return limit is None or area <= limit * image_area

    def _detect_fake(self, image: np.ndarray) -> List[Detection]:
        detections = []
        for object_type, color in fake_colors(self._prompts).items():
            color = np.array(color, dtype=np.int16)
            close = np.abs(image.astype(np.int16) - color).max(axis=-1) <= 8
            labels, num = ndimage.label(close)
            for region in range(1, num + 1):
                mask = labels == region
                if mask.sum() < FAKE_MIN_AREA:
                    continue
                ys, xs = np.nonzero(mask)
                box = (float(xs.min()), float(ys.min()), float(xs.max() + 1),
                       float(ys.max() + 1))
                if self._small_enough(object_type, box, mask.size):
                    detections.append(Detection(object_type, box, 1.0, mask))
        return detections

    def _load_models(self) -> None:
        # pylint: disable=import-outside-toplevel
        from transformers import AutoModelForZeroShotObjectDetection, \
            AutoProcessor, Sam2Model, Sam2Processor
        dino_path = weights.require("grounding_dino")
        sam_path = weights.require("sam2")
        self._dino_processor = AutoProcessor.from_pretrained(str(dino_path))
        self._dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(dino_path)).eval().to(self._device)
        self._sam_processor = Sam2Processor.from_pretrained(str(sam_path))
        self._sam = Sam2Model.from_pretrained(str(sam_path)).eval().to(
            self._device)

    def _detect_models(self, image: np.ndarray) -> List[Detection]:
        pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        labels = [p for prompts in self._prompts.values() for p in prompts]
        inputs = self._dino_processor(images=pil,
                                      text=[labels],
                                      return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._dino(**inputs)
        result = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil.size[::-1]])[0]
        found = []
        for box, score, text in zip(result["boxes"].tolist(),
                                    result["scores"].tolist(),
                                    result["text_labels"]):
            object_type = _type_of_label(text, self._prompts)
            if object_type is not None and self._small_enough(
                    object_type, tuple(box), pil.size[0] * pil.size[1]):
                found.append((object_type, tuple(box), float(score)))
        found = _suppress_duplicates(found)
        if not found:
            return []
        masks = self._masks(pil, [box for _, box, _ in found])
        return [
            Detection(object_type, box, score, mask)
            for (object_type, box, score), mask in zip(found, masks)
        ]

    def _masks(self, pil: Image.Image, boxes: List[Box]) -> List[np.ndarray]:
        """SAM2 masks of the boxes."""
        inputs = self._sam_processor(images=pil,
                                     input_boxes=[[list(b) for b in boxes]],
                                     return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._sam(**inputs, multimask_output=False)
        masks = self._sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        return [m[0].numpy().astype(bool) for m in masks]


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (union - inter) if union > inter else 0.0


def _suppress_duplicates(found: List[Tuple[str, Box, float]],
                         max_iou: float = 0.5,
                         max_iou_other: float = 0.7
                        ) -> List[Tuple[str, Box, float]]:
    """Keep the best scored of detections of one type that overlap more than
    max_iou (one toy matched by two of the type's prompts), and of
    detections of different types that overlap more than max_iou_other (the
    tray, found by its own prompt and taken for a toy)."""
    kept: List[Tuple[str, Box, float]] = []
    for detection in sorted(found, key=lambda d: -d[2]):
        if all(
                box_iou(d[1], detection[1]) <= (
                    max_iou if d[0] == detection[0] else max_iou_other)
                for d in kept):
            kept.append(detection)
    return kept


def _type_of_label(label: str,
                   prompts: Dict[str, List[str]]) -> Optional[str]:
    """The object type whose prompt GroundingDINO matched: GroundingDINO
    returns the words of the prompt that it matched."""
    label = label.strip().lower()
    exact = [t for t, ps in prompts.items() if label in map(str.lower, ps)]
    if exact:
        return exact[0]
    partial = {
        t
        for t, ps in prompts.items() for p in map(str.lower, ps)
        if label and (label in p or p in label)
    }
    return partial.pop() if len(partial) == 1 else None
