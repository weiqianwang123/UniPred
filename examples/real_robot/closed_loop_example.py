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
  in, its camera recording each skill for the tracker; the loop's camera
  images go to --frames_dir, and the end tells whether the goal holds in
  the simulator.

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
from unipred.loop_video import LoopVideo, predicate_labels


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


def _step(nsrt: _GroundNSRT) -> str:
    return f"{nsrt.option.name}({', '.join(o.name for o in nsrt.option_objs)})"


def run_closed_loop(deployment: Deployment,
                    robot: RobotInterface,
                    objects: Dict[str, str],
                    goal_atoms: Sequence[Tuple[str, Sequence[str]]],
                    max_steps: int,
                    video: Optional[LoopVideo] = None) -> bool:
    """Run the loop (see module doc); return whether the goal holds. The
    video, if any, receives every keyframe and skill."""
    plan: List[_GroundNSRT] = []
    done: List[str] = []
    executed: Optional[_GroundNSRT] = None
    previous = None
    for _ in range(max_steps + 1):
        frames = robot.video() if executed is not None else None
        if video is not None and executed is not None:
            video.skill(frames, done + [_step(n) for n in plan],
                        len(done) - 1)
        image = robot.capture()
        state, observation = deployment.observe(image,
                                                objects,
                                                previous,
                                                frames=frames)
        previous = observation
        goal = deployment.goal(goal_atoms, state)
        note = ""
        if all(a.holds(state) for a in goal):
            logging.info("The goal holds.")
            if video is not None:
                video.keyframe(image, observation, objects, state, done,
                               len(done), "the goal holds")
            return True
        if executed is not None and not _effects_happened(executed, state):
            logging.info(f"The effects of {executed.name} did not happen; "
                         "replanning.")
            note = f"{_step(executed)} did not work: replanning"
            plan = []
        if not plan:
            try:
                plan = deployment.plan(state, goal)
            except (ApproachFailure, ApproachTimeout) as err:
                logging.info(f"No plan: {err}")
                return False
            logging.info(f"Plan: {', '.join(_step(n) for n in plan)}")
        if not plan:
            return False
        if video is not None:
            video.keyframe(image, observation, objects, state,
                           done + [_step(n) for n in plan], len(done), note)
        executed = plan.pop(0)
        done.append(_step(executed))
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
    parser.add_argument("--video",
                        help="an annotated video of the loop to write "
                        "(.mp4 or .gif; see unipred/loop_video.py)")
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
    video = None
    if args.video:
        predicates = deployment.predicates()
        invented = [p for p in predicates if p not in deployment.env.predicates]
        video = LoopVideo(
            f"{os.path.basename(os.path.normpath(args.session))}: "
            f"{deployment.approach.get_name()} in the closed loop",
            predicates,
            predicate_labels(invented, sorted(deployment.env.predicates),
                             deployment.env.annotated_states()))
    if args.robot == "replay":
        run_closed_loop(deployment, FakeRobotInterface(args.session),
                        session["objects"], session["goal"], args.max_steps,
                        video)
        if video is not None:
            video.save(args.video)
        return
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    from unipred.robot_sim.cluttered_retrieval_mujoco import \
        ClutteredRetrievalMujoco
    from unipred.robot_sim.franka_scene import MujocoRobot
    from unipred.robot_sim.table_clean_mujoco import TableCleanMujoco
    if "sim" not in session:
        raise ValueError(f"{args.session} was not recorded in the simulator "
                         "(its session.json has no \"sim\").")
    scene = {
        "table_clean": TableCleanMujoco,
        "cluttered_retrieval": ClutteredRetrievalMujoco,
    }[session.get("domain", "table_clean")]
    sim = scene.from_description(session["sim"])
    robot = MujocoRobot(sim)
    run_closed_loop(deployment, robot, session["objects"], session["goal"],
                    args.max_steps, video)
    if video is not None:
        video.save(args.video)
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
