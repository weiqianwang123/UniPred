"""Generate the synthetic fixture: recorded sessions of Table Clean in the
raw format of examples/real_robot/README.md.

The sessions are oracle demonstrations of Table Clean Sim (train sessions)
and initial scenes of its out-of-distribution test tasks (test scenes),
drawn as top-down images in which each object is a region of the colour
that the fake detector associates with its type (the types of the prompts
in configs/real_robot.yaml). The whole chain then runs without network
weights.

    python examples/real_robot/fixture/make_fixture.py
"""

import argparse
import json
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import GroundAtom, Object, State
import unipred  # pylint: disable=unused-import
from unipred.perception.detector import Detector, fake_colors

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..", "..", "..")
_SIZE = (160, 140)
_BACKGROUND = (128, 128, 128)
_DIRT = (110, 80, 50)
_DIRT_SPOTS = [(0.3, 0.55), (0.45, 0.3), (0.2, 0.8)]
# Half sizes in pixels.
_HALF = {"robot": (5, 4), "toy": (4, 4), "wiper": (6, 3), "box": (13, 13)}


def _pixel(x: float, y: float) -> Tuple[int, int]:
    """Table coordinates ([0, 1] x [-0.15, 1]) to pixels, y pointing away
    from the robot."""
    width, height = _SIZE
    return (int(round(10 + x * (width - 20))),
            int(round(height - 5 - (y + 0.15) / 1.15 * (height - 10))))


def _render(
    state: State, colors: Dict[str, Tuple[int, int, int]]
) -> Tuple[Image.Image, Dict[str, List[float]]]:
    """The top-down image of a state and the box drawn for each object."""
    image = Image.new("RGB", _SIZE, _BACKGROUND)
    boxes: Dict[str, List[float]] = {}
    draw = ImageDraw.Draw(image)
    by_type: Dict[str, List[Object]] = {}
    for obj in sorted(state, key=lambda o: o.name):
        by_type.setdefault(obj.type.name, []).append(obj)
    table = by_type["table"][0]
    (x0, y1), (x1, y0) = _pixel(0, 0), _pixel(1, 1)
    draw.rectangle([x0, y0, x1, y1], fill=colors["table"])
    boxes[table.name] = [x0, y0, x1 + 1, y1 + 1]
    if state.get(table, "is_clean") < 0.5:
        for x, y in _DIRT_SPOTS:
            px, py = _pixel(x, y)
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=_DIRT)

    def _rect(obj: Object, at: Object, dy: int = 0) -> None:
        px, py = _pixel(state.get(at, "x"), state.get(at, "y"))
        hx, hy = _HALF[obj.type.name]
        draw.rectangle([px - hx, py + dy - hy, px + hx, py + dy + hy],
                       fill=colors[obj.type.name])
        boxes[obj.name] = [px - hx, py + dy - hy, px + hx + 1, py + dy + hy + 1]

    for box in by_type["box"]:
        _rect(box, box)
    robot = by_type["robot"][0]
    held = []
    for obj in by_type["toy"] + by_type["wiper"]:
        if state.get(obj, "held") > 0.5:
            held.append(obj)
        else:
            _rect(obj, obj)
    _rect(robot, robot)
    # A held object is drawn just beyond the gripper.
    for obj in held:
        _rect(obj, robot, dy=-9)
    return image, boxes


def _boxes_to_give(image: Image.Image, boxes: Dict[str, List[float]],
                   objects: Dict[str, str],
                   prompts: Dict[str, str]) -> Dict[str, List[float]]:
    """The boxes of the objects whose type the fake detector does not count
    right in the image (overlapping objects of a type merge into one
    region): the recording gives them instead."""
    counts: Dict[str, int] = {}
    for detection in Detector("fake", prompts).detect(np.asarray(image)):
        counts[detection.object_type] = counts.get(detection.object_type,
                                                   0) + 1
    given = {}
    for object_type in set(objects.values()):
        names = [n for n, t in objects.items() if t == object_type]
        if counts.get(object_type, 0) != len(names):
            given.update({n: boxes[n] for n in names})
    return given


def _atoms(atoms: List[GroundAtom]) -> List:
    return sorted([a.predicate.name, [o.name for o in a.objects]]
                  for a in atoms)


def _dumps(content: object) -> str:
    """JSON with one line per list entry or mapping item."""
    if isinstance(content, list) and content:
        return "[\n" + ",\n".join(" " + json.dumps(c)
                                   for c in content) + "\n]"
    if isinstance(content, dict):
        return "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}"
                                   for k, v in content.items()) + "\n}"
    return json.dumps(content)


def main() -> None:
    """Write the fixture (see module doc)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--out", default=os.path.join(_HERE, "raw"))
    parser.add_argument("--num_train", type=int, default=4)
    parser.add_argument("--num_test", type=int, default=2)
    args = parser.parse_args()
    with open(os.path.join(_REPO, "configs", "real_robot.yaml"),
              encoding="utf-8") as f:
        prompts = yaml.safe_load(f)["prompts"]
    colors = fake_colors(prompts)
    utils.reset_config({
        "env": "table_clean_sim",
        "num_train_tasks": args.num_train,
        "num_test_tasks": args.num_test,
        "sesame_task_planner": "astar",
        "data_dir": tempfile.mkdtemp(),
    })
    env = get_or_create_env("table_clean_sim")
    derived = {p.name for p in env.derived_predicates}
    predicates = {p for p in env.predicates if p.name not in derived}
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    sessions = []
    for k, traj in enumerate(dataset.trajectories):
        keyframes = [0] + [
            i for i in range(1, len(traj.actions))
            if traj.actions[i].get_option() is not traj.actions[i -
                                                                1].get_option()
        ] + [len(traj.actions)]
        options = [traj.actions[i].get_option() for i in keyframes[:-1]]
        task = train_tasks[traj.train_task_idx]
        sessions.append((f"train_{k:02d}", "train",
                         [traj.states[i] for i in keyframes], options,
                         task.goal))
    for k, env_task in enumerate(env.get_test_tasks()):
        task = env_task.task
        sessions.append((f"test_{k:02d}", "test", [task.init], [], task.goal))
    for name, split, states, options, goal in sessions:
        directory = os.path.join(args.out, name)
        os.makedirs(directory, exist_ok=True)
        objects = {o.name: o.type.name for o in sorted(states[0])}
        given = []
        for i, state in enumerate(states):
            image, boxes = _render(state, colors)
            image.save(os.path.join(directory, f"{i}.png"))
            given.append(_boxes_to_give(image, boxes, objects, prompts))
        session = {
            "split": split,
            "objects": {o.name: o.type.name
                        for o in sorted(states[0])},
            "goal": _atoms(list(goal)),
        }
        actions = [{
            "option": o.name,
            "objects": [obj.name for obj in o.objects]
        } for o in options]
        atoms = [_atoms(list(utils.abstract(s, predicates))) for s in states]
        files = [("session.json", session), ("actions.json", actions),
                 ("atoms.json", atoms)]
        if any(given):
            files.append(("boxes.json", given))
        for file_name, content in files:
            with open(os.path.join(directory, file_name), "w",
                      encoding="utf-8") as f:
                f.write(_dumps(content) + "\n")
    print(f"Wrote {len(sessions)} sessions to {args.out}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
