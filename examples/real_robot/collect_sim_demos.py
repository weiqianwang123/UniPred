"""Collect recorded sessions of Table Clean in the MuJoCo simulator, in the
raw format of README.md, as a stand-in for demonstrations on a real robot.

Tasks come from Table Clean Sim: train sessions start from its training
distribution (two toys on the table, the sponge in the tray near the robot)
and are demonstrated by its oracle plan, executed skill by skill by the
simulated Franka (unipred/robot_sim/table_clean_mujoco.py) with a keyframe
image before each skill and after the last; test scenes are the initial
scenes of its out-of-distribution tasks (three toys, toys already in the
tray, a toy or the sponge in hand, the tray on the far side). Every keyframe
records the ground-truth atoms (atoms.json) and the objects' image boxes
(gt_boxes.json, for evaluating detection; the processing does not read
it). With --give_boxes, the boxes are also written as boxes.json, which
replaces detection.

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
import json
import logging
import os
import tempfile
from typing import List

from PIL import Image

from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from unipred.robot_sim.table_clean_mujoco import SimulationError, \
    TableCleanMujoco

_ATTEMPTS = 3


def _dumps(content: object) -> str:
    """JSON with one line per list entry or mapping item."""
    if isinstance(content, list) and content:
        return "[\n" + ",\n".join(" " + json.dumps(c)
                                  for c in content) + "\n]"
    if isinstance(content, dict):
        return "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}"
                                  for k, v in content.items()) + "\n}"
    return json.dumps(content)


def _write(directory: str, sim: TableCleanMujoco, split: str, goal: List,
           steps: List, give_boxes: bool) -> None:
    """Execute the steps in the scene and write the session (nothing if a
    step fails)."""
    images, atoms, boxes = [], [], []

    def _keyframe() -> None:
        images.append(Image.fromarray(sim.render()))
        atoms.append([[name, list(args)] for name, args in sim.atoms()])
        boxes.append(
            {n: [round(v, 1) for v in box]
             for n, box in sim.boxes().items()})

    _keyframe()
    for option_name, objects in steps:
        if not sim.execute(option_name, objects):
            raise RuntimeError(f"{option_name}{tuple(objects)} does not apply "
                               f"in {directory}.")
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
    files = {
        "session.json": {
            "split": split,
            "objects": sim.objects(),
            "goal": [[name, list(args)] for name, args in goal],
            # Not read by the processing: which scanned toy each toy is,
            # and the scene to rebuild (closed_loop_example.py --robot sim).
            "toy_models": sim.toy_models,
            "sim": sim.description(),
        },
        "actions.json": [{
            "option": name,
            "objects": list(objects)
        } for name, objects in steps],
        "atoms.json": atoms,
        "gt_boxes.json": boxes,
    }
    if give_boxes:
        files["boxes.json"] = boxes
    os.makedirs(directory, exist_ok=True)
    for index, image in enumerate(images):
        image.save(os.path.join(directory, f"{index}.png"))
    for name, content in files.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(_dumps(content) + "\n")


def main() -> None:
    """Collect the sessions (see module doc)."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--give_boxes", action="store_true",
                        help="write the ground-truth boxes as boxes.json")
    parser.add_argument("--grasps",
                        choices=["graspgenx", "top_down"],
                        default="graspgenx")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    utils.reset_config({
        "env": "table_clean_sim",
        "seed": args.seed,
        "num_train_tasks": args.num_train,
        "num_test_tasks": args.num_test,
        "sesame_task_planner": "astar",
        "data_dir": tempfile.mkdtemp(),
    })
    env = get_or_create_env("table_clean_sim")
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
            sim = TableCleanMujoco.from_sim_state(task.init,
                                                  seed=seed + 100000 * attempt,
                                                  grasps=args.grasps)
            try:
                _write(os.path.join(args.out, name), sim, split, goal, steps,
                       args.give_boxes)
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
