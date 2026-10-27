"""Closed-loop deployment of a model learned on a real domain.

The loop perceives the scene, plans with the learned operators, executes
the first skill of the plan, perceives again (re-grounding the atoms), and
replans when the skill's expected effects (its operator's add and delete
effects) did not happen; otherwise it executes the next skill of the plan.
It ends when the goal holds, no plan is found or the step budget is spent.

A robot implements RobotInterface: capture() returns an RGB image of the
scene, execute(option_name, objects) runs a skill (e.g. a motion-planned
pick or a learned policy) on the named objects. Two come with the example:

- replay (FakeRobotInterface) replays a recorded session: capture() returns
  the current keyframe and execute() moves on to the next one (whatever the
  skill), so the example runs offline;
- sim (unipred/robot_sim, MujocoRobot) runs the skills on the simulated
  Franka, from the scene a session of collect_sim_demos.py was recorded
  in; the loop's camera images go to --frames_dir, and the end tells
  whether the goal holds in the simulator.

    python examples/real_robot/closed_loop_example.py \\
        --session examples/real_robot/fixture/raw/train_00 \\
        --detector fake --encoder fake \\
        --config configs/table_clean_real/unipred.yaml --seed 0 \\
        --approach_dir outputs/table_clean_real/unipred/seed0/approach \\
        --table_clean_real_data_dir outputs/table_clean_real/processed
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.structs import State, _GroundNSRT
from unipred.deploy import Deployment, RobotInterface, load_image


class FakeRobotInterface(RobotInterface):
    """Replays the keyframes of a recorded session."""

    def __init__(self, session_dir: str) -> None:
        self._images: List[str] = []
        while os.path.exists(
                os.path.join(session_dir, f"{len(self._images)}.png")):
            self._images.append(
                os.path.join(session_dir, f"{len(self._images)}.png"))
        self._index = 0

    def capture(self) -> np.ndarray:
        return load_image(self._images[self._index])

    def execute(self, option_name: str, objects: Sequence[str]) -> None:
        logging.info(f"Robot executes {option_name}({', '.join(objects)})")
        self._index = min(self._index + 1, len(self._images) - 1)


def _effects_happened(nsrt: _GroundNSRT, state: State) -> bool:
    return all(a.holds(state) for a in nsrt.add_effects) and not any(
        a.holds(state) for a in nsrt.delete_effects)


def run_closed_loop(deployment: Deployment, robot: RobotInterface,
                    objects: Dict[str, str],
                    goal_atoms: Sequence[Tuple[str, Sequence[str]]],
                    max_steps: int) -> bool:
    """Run the loop (see module doc); return whether the goal holds."""
    plan: List[_GroundNSRT] = []
    executed: Optional[_GroundNSRT] = None
    previous = None
    for _ in range(max_steps + 1):
        state, observation = deployment.observe(robot.capture(), objects,
                                                previous)
        previous = observation
        goal = deployment.goal(goal_atoms, state)
        if all(a.holds(state) for a in goal):
            logging.info("The goal holds.")
            return True
        if executed is not None and not _effects_happened(executed, state):
            logging.info(f"The effects of {executed.name} did not happen; "
                         "replanning.")
            plan = []
        if not plan:
            try:
                plan = deployment.plan(state, goal)
            except (ApproachFailure, ApproachTimeout) as err:
                logging.info(f"No plan: {err}")
                return False
            steps = ", ".join(
                f"{n.option.name}({', '.join(o.name for o in n.option_objs)})"
                for n in plan)
            logging.info(f"Plan: {steps}")
        if not plan:
            return False
        executed = plan.pop(0)
        robot.execute(executed.option.name,
                      [o.name for o in executed.option_objs])
    logging.info("The step budget is spent.")
    return False


def main() -> None:
    """Run the loop on a replayed session (see module doc)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--session",
                        required=True,
                        help="a recorded session folder (raw format)")
    parser.add_argument("--robot", choices=["replay", "sim"], default="replay")
    parser.add_argument("--frames_dir",
                        help="with --robot sim: where the camera images go")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--robot_config", default="configs/real_robot.yaml")
    parser.add_argument("--detector", help="override the config's detector")
    parser.add_argument("--encoder", help="override the config's encoder")
    args, training_flags = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(os.path.join(args.session, "session.json"),
              encoding="utf-8") as f:
        session = json.load(f)
    deployment = Deployment(training_flags, args.robot_config, args.detector,
                            args.encoder)
    if args.robot == "replay":
        run_closed_loop(deployment, FakeRobotInterface(args.session),
                        session["objects"], session["goal"], args.max_steps)
        return
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    from unipred.robot_sim.table_clean_mujoco import MujocoRobot, \
        TableCleanMujoco
    if "sim" not in session:
        raise ValueError(f"{args.session} was not recorded in the simulator "
                         "(its session.json has no \"sim\").")
    sim = TableCleanMujoco.from_description(session["sim"])
    robot = MujocoRobot(sim)
    run_closed_loop(deployment, robot, session["objects"], session["goal"],
                    args.max_steps)
    goal = {(name, tuple(args_)) for name, args_ in session["goal"]}
    reached = goal <= set(sim.atoms())
    logging.info("In the simulator, the goal "
                 f"{'holds' if reached else 'does not hold'}.")
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        for k, frame in enumerate(robot.frames):
            Image.fromarray(frame).save(
                os.path.join(args.frames_dir, f"{k}.png"))
    sim.close()


if __name__ == "__main__":
    main()
