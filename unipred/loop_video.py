"""An annotated video of the closed loop (examples/real_robot/
closed_loop_example.py --video): what the robot sees, what it believes and
what it does.

Each keyframe shows the camera image with the box and name of every object
that perception found, beside a panel with the plan (the skills done, the
one about to run, the rest) and the atoms true in the perceived state.
An invented predicate is named after the hidden ground-truth predicate,
or its negation, that it agrees with most on the annotated keyframes of the
training sessions (with the agreement: green from 90 percent), so that what
UniPred discovered can be read; one that matches none on at least 60
percent is left out. Given predicates keep their names. The frames
recorded while a skill runs follow its keyframe, with the skill
highlighted in the plan.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import imageio
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from predicators import utils
from predicators.structs import GroundAtom, Predicate, State
from unipred.perception.scene import SceneObservation

_PANEL_WIDTH = 520
_HEADER = 40
_COLORS = [(230, 60, 60), (40, 120, 230), (40, 170, 80), (230, 150, 20),
           (150, 60, 200), (20, 170, 180), (200, 80, 150), (120, 120, 120)]
_MIN_AGREEMENT = 0.9  # shown in green from here
_MIN_SHOWN = 0.6  # not shown below


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")),
        size)


def predicate_labels(
        invented: Sequence[Predicate], ground_truth: Sequence[Predicate],
        states: Sequence[State]) -> Dict[Predicate, Tuple[str, float]]:
    """For each invented predicate, the ground-truth predicate of the same
    argument types, or its negation ("¬Name"), that it agrees with most on
    the groundings of the annotated states (whose simulator states keep the
    true atoms), and the agreement."""
    agree: Dict[Tuple[Predicate, Predicate], List[int]] = {}
    for state in states:
        truth = state.simulator_state.atoms
        believed = utils.abstract(state, set(invented))
        for pred in invented:
            for gt in ground_truth:
                if list(gt.types) != list(pred.types):
                    continue
                count = agree.setdefault((pred, gt), [0, 0])
                domains = [state.get_objects(t) for t in pred.types]
                for args in itertools.product(*domains):
                    if len(set(args)) < len(args):
                        continue
                    count[0] += (GroundAtom(pred, list(args)) in believed) \
                        == ((gt.name, tuple(o.name for o in args)) in truth)
                    count[1] += 1
    labels = {}
    for pred in invented:
        scored = [(c[0] / c[1], gt.name)
                  for (p, gt), c in agree.items() if p == pred and c[1]]
        scored += [(1 - score, f"¬{name}") for score, name in scored]
        if scored:
            # On a tie, a predicate rather than a negation.
            score, name = max(scored,
                              key=lambda s: (s[0], not s[1].startswith("¬")))
            labels[pred] = (name, score)
    return labels


class LoopVideo:
    """The frames of the video (see module doc)."""

    def __init__(self,
                 title: str,
                 predicates: Set[Predicate],
                 labels: Mapping[Predicate, Tuple[str, float]],
                 fps: int = 10,
                 hold: float = 1.5) -> None:
        self._title = title
        self._predicates = predicates
        self._labels = dict(labels)
        self._fps = fps
        self._hold = hold
        self._frames: List[np.ndarray] = []
        self._atoms: List[Tuple[str, Tuple[int, int, int]]] = []
        self._colors: Dict[str, Tuple[int, int, int]] = {}

    def keyframe(self,
                 image: np.ndarray,
                 observation: SceneObservation,
                 objects: Mapping[str, str],
                 state: State,
                 plan: Sequence[str],
                 current: int,
                 note: str = "") -> None:
        """A keyframe: its image with the objects' boxes, the plan (steps
        before current are done) and the atoms true in its state."""
        shown = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(shown)
        font = _font(14)
        area = image.shape[0] * image.shape[1]
        for name, box in sorted(observation.boxes.items()):
            if (box[2] - box[0]) * (box[3] - box[1]) > 0.5 * area:
                continue  # the table, say
            color = self._color(objects[name])
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, max(0, box[1] - 17)),
                      name,
                      fill=color,
                      font=font,
                      stroke_width=2,
                      stroke_fill=(255, 255, 255))
        self._atoms = self._true_atoms(state)
        frame = self._compose(np.asarray(shown), plan, current, note)
        self._frames.extend([frame] * max(1, int(self._hold * self._fps)))

    def skill(self, frames: Optional[Sequence[np.ndarray]],
              plan: Sequence[str], current: int) -> None:
        """The frames recorded while plan[current] ran."""
        for frame in frames or []:
            self._frames.append(
                self._compose(frame, plan, current, "executing"))

    @property
    def frames(self) -> List[np.ndarray]:
        """The frames so far."""
        return list(self._frames)

    def save(self, path: str) -> None:
        """Write the video (mp4, or gif by the file's extension)."""
        if path.endswith(".gif"):
            imageio.mimwrite(path,
                             self._frames,
                             duration=1000 / self._fps,
                             loop=0)
        else:
            imageio.mimwrite(path, self._frames, fps=self._fps)

    def _true_atoms(self,
                    state: State) -> List[Tuple[str, Tuple[int, int, int]]]:
        """The atoms true in a state, as lines and their colours."""
        lines = []
        for atom in sorted(utils.abstract(state, self._predicates)):
            args = ", ".join(o.name for o in atom.objects)
            if atom.predicate in self._labels:
                name, score = self._labels[atom.predicate]
                if score < _MIN_SHOWN:
                    continue
                color = (20, 120, 60) if score >= _MIN_AGREEMENT else \
                    (200, 120, 20)
                lines.append(
                    (f"≈{name}({args})  invented, {score:.0%}", color))
            elif atom.predicate.name.startswith(("neural_", "forall_",
                                                 "exists_")):
                continue  # invented, but like no ground-truth predicate
            else:
                lines.append((f"{atom.predicate.name}({args})", (40, 40, 40)))
        return lines

    def _color(self, object_type: str) -> Tuple[int, int, int]:
        if object_type not in self._colors:
            self._colors[object_type] = _COLORS[len(self._colors) %
                                                len(_COLORS)]
        return self._colors[object_type]

    def _compose(self, image: np.ndarray, plan: Sequence[str], current: int,
                 note: str) -> np.ndarray:
        height, width = image.shape[:2]
        canvas = Image.new("RGB", (width + _PANEL_WIDTH, height + _HEADER),
                           (250, 250, 248))
        canvas.paste(Image.fromarray(image), (0, _HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width + _PANEL_WIDTH, _HEADER),
                       fill=(35, 40, 48))
        draw.text((12, 9), self._title, fill=(255, 255, 255), font=_font(18))
        if note:
            draw.text((width + 12, 11),
                      note,
                      fill=(255, 210, 90),
                      font=_font(15))
        x, y = width + 14, _HEADER + 10
        bold, small = _font(16), _font(13)
        draw.text((x, y), "Plan", fill=(30, 30, 30), font=bold)
        y += 24
        for k, step in enumerate(plan):
            if k < current:
                mark, color = "✓", (130, 130, 130)
            elif k == current:
                mark, color = "▶", (200, 60, 20)
            else:
                mark, color = "·", (40, 40, 40)
            draw.text((x, y), f"{mark} {step}", fill=color, font=small)
            y += 18
            if y > _HEADER + height * 0.45:
                break
        y += 10
        draw.text((x, y), "What the robot believes", fill=(30, 30, 30),
                  font=bold)
        y += 24
        for line, color in self._atoms:
            draw.text((x, y), line, fill=color, font=small)
            y += 17
            if y > _HEADER + height - 18:
                break
        return np.asarray(canvas)
