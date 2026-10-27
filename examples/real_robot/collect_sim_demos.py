"""Collect recorded sessions of a real-robot domain in the MuJoCo simulator,
in the raw format of README.md, as a stand-in for demonstrations on a real
robot: Table Clean (--domain table_clean, the default) or Cluttered
Retrieval (--domain cluttered_retrieval).

Tasks come from the domain's simulated counterpart (Table Clean Sim,
Cluttered Retrieval Sim): train sessions start from its training
distribution and are demonstrated by its oracle plan, executed skill by
skill by the simulated Franka (unipred/robot_sim/table_clean_mujoco.py,
cluttered_retrieval_mujoco.py) with a keyframe image before each skill and
after the last; test scenes are the initial scenes of its
out-of-distribution tasks (Table Clean: three toys, toys already in the
tray, a toy or the sponge in hand, the tray on the far side; Cluttered
Retrieval: three targets, one behind two obstacles). Numbered objects (toys,
targets, obstacles) are named left to right in the first keyframe, as the
processing expects. Every
keyframe records the ground-truth atoms (atoms.json) and the objects' image
boxes (gt_boxes.json, for evaluating detection, or for the boxes of objects
the detector misses: process_demos.py --fill_missing gt_boxes.json). With
--give_boxes, the boxes are also written as boxes.json, which replaces
detection. The camera records every skill, --video_hz frames per simulated
second (frames/<k>/<j>.jpg during skill k), for the tracker to follow the
objects from keyframe to keyframe.

Picks take their grasps from GraspGenX (CDG's perception server, see
unipred/robot_sim/graspgenx.py), or with --grasps top_down straight from
above. A session whose skill cannot be carried out (no grasp closes, say)
or ends with the objects not where the atoms say is simulated again with
another seed, up to three times.

    python -m unipred.robot_sim.assets   # once: the robot and the objects
    python examples/real_robot/collect_sim_demos.py \\
        --out outputs/table_clean_mujoco/raw --num_train 20 --num_test 5
"""

import argparse
import io
import json
import logging
import os
import tempfile
import re
from typing import Any, Dict, List

from PIL import Image

from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from unipred.robot_sim.cluttered_retrieval_mujoco import \
    ClutteredRetrievalMujoco
from unipred.robot_sim.franka_scene import FrankaTableScene, SimulationError
from unipred.robot_sim.table_clean_mujoco import TableCleanMujoco

_ATTEMPTS = 3
# Domain -> its simulated counterpart and its scene.
_DOMAINS = {
    "table_clean": ("table_clean_sim", TableCleanMujoco),
    "cluttered_retrieval":
    ("cluttered_retrieval_sim", ClutteredRetrievalMujoco),
}


def _dumps(content: object) -> str:
    """JSON with one line per list entry or mapping item."""
    if isinstance(content, list) and content:
        return "[\n" + ",\n".join(" " + json.dumps(c)
                                  for c in content) + "\n]"
    if isinstance(content, dict):
        return "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}"
                                  for k, v in content.items()) + "\n}"
    return json.dumps(content)


def _jpeg(frame) -> bytes:
    with io.BytesIO() as f:
        Image.fromarray(frame).save(f, "JPEG", quality=90)
        return f.getvalue()


def _renamed(content: Any, rename: Dict[str, str]) -> Any:
    """Content (JSON) with every string that is an old name renamed."""
    if isinstance(content, str):
        return rename.get(content, content)
    if isinstance(content, (list, tuple)):
        return [_renamed(c, rename) for c in content]
    if isinstance(content, dict):
        return {_renamed(k, rename): _renamed(v, rename)
                for k, v in content.items()}
    return content


def _write(directory: str, domain: str, sim: FrankaTableScene, split: str,
           goal: List, steps: List, give_boxes: bool, video_hz: float) -> None:
    """Execute the steps in the scene and write the session (nothing if a
    step fails)."""
    images, atoms, boxes, videos = [], [], [], []

    def _keyframe() -> None:
        images.append(Image.fromarray(sim.render()))
        atoms.append([[name, list(args)] for name, args in sim.atoms()])
        boxes.append(
            {n: [round(v, 1) for v in box]
             for n, box in sim.boxes().items()})

    _keyframe()
    if video_hz > 0:
        sim.video_period = 1.0 / video_hz
    for option_name, objects in steps:
        sim.video = [] if video_hz > 0 else None
        if not sim.execute(option_name, objects):
            raise RuntimeError(f"{option_name}{tuple(objects)} does not apply "
                               f"in {directory}.")
        videos.append([_jpeg(frame) for frame in sim.video or []])
        sim.video = None
        problems = sim.problems()
        if problems:
            raise SimulationError(f"After {option_name}{tuple(objects)}: "
                                  f"{'; '.join(problems)}.")
        _keyframe()
    if split == "train" and not set(map(tuple, goal)) <= {
            (n, tuple(a))
            for n, a in atoms[-1]
    }:
        raise RuntimeError(f"The demonstration in {directory} does not reach "
                           "the goal.")
    # The raw format names the objects of a type left to right in the first
    # keyframe (by their boxes' centres): rename the numbered ones so.
    rename: Dict[str, str] = {}
    for object_type in set(sim.objects().values()):
        numbered = [n for n, t in sim.objects().items()
                    if t == object_type and re.fullmatch(
                        re.escape(object_type) + r"\d+", n)]
        order = sorted(numbered,
                       key=lambda n: (boxes[0][n][0] + boxes[0][n][2],
                                      boxes[0][n][1] + boxes[0][n][3]))
        rename.update(
            {old: f"{object_type}{k}" for k, old in enumerate(order)})
    files = {
        "session.json": {
            "split": split,
            "objects": dict(sorted(_renamed(sim.objects(), rename).items())),
            "goal": sorted(_renamed(goal, rename)),
            # Not read by the processing: the domain, the objects' scanned
            # models, and the scene to rebuild (closed_loop_example.py
            # --robot sim).
            "domain": domain,
            "models": dict(
                sorted(_renamed(sim.scanned_models(), rename).items())),
            "sim": _renamed(sim.description(), rename),
        },
        "actions.json": [{
            "option": name,
            "objects": _renamed(list(objects), rename)
        } for name, objects in steps],
        "atoms.json": [sorted(_renamed(a, rename)) for a in atoms],
        "gt_boxes.json": [
            dict(sorted(_renamed(frame, rename).items())) for frame in boxes
        ],
    }
    if give_boxes:
        files["boxes.json"] = files["gt_boxes.json"]
    os.makedirs(directory, exist_ok=True)
    for index, image in enumerate(images):
        image.save(os.path.join(directory, f"{index}.png"))
    for index, video in enumerate(videos if video_hz > 0 else []):
        os.makedirs(os.path.join(directory, "frames", str(index)))
        for j, frame in enumerate(video):
            with open(os.path.join(directory, "frames", str(index),
                                   f"{j}.jpg"), "wb") as f:
                f.write(frame)
    for name, content in files.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(_dumps(content) + "\n")


def main() -> None:
    """Collect the sessions (see module doc)."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--domain", choices=sorted(_DOMAINS),
                        default="table_clean")
    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--give_boxes", action="store_true",
                        help="write the ground-truth boxes as boxes.json")
    parser.add_argument("--grasps",
                        choices=["graspgenx", "top_down"],
                        default="graspgenx")
    parser.add_argument("--video_hz",
                        type=float,
                        default=5.0,
                        help="frames per simulated second recorded during "
                        "the skills (0: none)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    env_name, scene = _DOMAINS[args.domain]
    utils.reset_config({
        "env": env_name,
        "seed": args.seed,
        "num_train_tasks": args.num_train,
        "num_test_tasks": args.num_test,
        "sesame_task_planner": "astar",
        "data_dir": tempfile.mkdtemp(),
    })
    env = get_or_create_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    sessions = []
    for k, traj in enumerate(dataset.trajectories):
        steps = []
        for i, action in enumerate(traj.actions):
            option = action.get_option()
            if i == 0 or option is not traj.actions[i - 1].get_option():
                steps.append(
                    (option.name, [o.name for o in option.objects]))
        sessions.append((f"train_{k:02d}", "train",
                         train_tasks[traj.train_task_idx], steps,
                         args.seed * 1000 + k))
    for k, env_task in enumerate(env.get_test_tasks()):
        sessions.append((f"test_{k:02d}", "test", env_task.task, [],
                         args.seed * 1000 + 500 + k))
    for name, split, task, steps, seed in sessions:
        goal = sorted((a.predicate.name, tuple(o.name for o in a.objects))
                      for a in task.goal)
        for attempt in range(_ATTEMPTS):
            sim = scene.from_sim_state(task.init,
                                                  seed=seed + 100000 * attempt,
                                                  grasps=args.grasps)
            try:
                _write(os.path.join(args.out, name), args.domain, sim, split,
                       goal, steps, args.give_boxes, args.video_hz)
                break
            except SimulationError as e:
                logging.warning(f"{name}, attempt {attempt + 1}: {e}")
            finally:
                sim.close()
        else:
            raise RuntimeError(f"{name} failed {_ATTEMPTS} times.")
        logging.info(f"{name}: {len(steps)} skills" if steps else
                     f"{name}: initial scene")
    logging.info(f"Wrote {args.num_train} train sessions and "
                 f"{args.num_test} test scenes to {args.out}.")


if __name__ == "__main__":
    main()
