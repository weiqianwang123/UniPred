"""Object detection: boxes for the objects of each type from a text prompt
per type (GroundingDINO), refined into masks (SAM2).

Backends:

- "groundingdino_sam2": GroundingDINO and SAM2 through transformers, from
  their installed weights (unipred/perception/weights.py);
- "fake": no network weights, for tests and the synthetic fixture of
  examples/real_robot. The objects of each type are drawn as regions of the
  colour fake_colors() gives the type; the detector returns every connected
  region of that colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from unipred.perception import weights

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


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
    """Detects the objects of the types in prompts (type -> text prompt)."""

    def __init__(self,
                 backend: str,
                 prompts: Mapping[str, str],
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 device: str = "cpu") -> None:
        if backend not in ("groundingdino_sam2", "fake"):
            raise ValueError(f"Unknown detector backend: {backend!r} (use "
                             "groundingdino_sam2 or fake).")
        self.backend = backend
        self._prompts = dict(prompts)
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._device = torch.device(device if torch.cuda.is_available()
                                    or device == "cpu" else "cpu")
        if backend == "groundingdino_sam2":
            self._load_models()

    def detect(self, image: np.ndarray) -> List[Detection]:
        """The detections in an RGB image (H x W x 3, uint8)."""
        if self.backend == "fake":
            return self._detect_fake(image)
        return self._detect_models(image)

    def _detect_fake(self, image: np.ndarray) -> List[Detection]:
        detections = []
        for object_type, color in fake_colors(self._prompts).items():
            color = np.array(color, dtype=np.int16)
            close = np.abs(image.astype(np.int16) - color).max(axis=-1) <= 8
            labels, num = ndimage.label(close)
            for region in range(1, num + 1):
                mask = labels == region
                ys, xs = np.nonzero(mask)
                box = (float(xs.min()), float(ys.min()), float(xs.max() + 1),
                       float(ys.max() + 1))
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
        types = list(self._prompts)
        labels = [self._prompts[t] for t in types]
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
            if object_type is not None:
                found.append((object_type, tuple(box), float(score)))
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


def _type_of_label(label: str, prompts: Dict[str, str]) -> Optional[str]:
    """The object type whose prompt GroundingDINO matched: GroundingDINO
    returns the words of the prompt that it matched."""
    label = label.strip().lower()
    exact = [t for t, p in prompts.items() if p.lower() == label]
    if exact:
        return exact[0]
    partial = [
        t for t, p in prompts.items() if label and (label in p.lower()
                                                     or p.lower() in label)
    ]
    return partial[0] if len(partial) == 1 else None
