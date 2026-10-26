"""Plan from one image with a model learned on a real domain: detection,
crop features, the learned classifiers ground the atoms, and the learned
operators give the plan (see unipred/deploy.py).

The scene is given by a session folder in the raw format (its first
keyframe, objects and goal), or by --image, --objects and --goal. All other
flags are those of the training run.

    python examples/real_robot/plan_from_image.py \\
        --session examples/real_robot/fixture/raw/test_00 \\
        --detector fake --encoder fake \\
        --config configs/table_clean_real/unipred.yaml --seed 0 \\
        --approach_dir outputs/table_clean_real/unipred/seed0/approach \\
        --table_clean_real_data_dir outputs/table_clean_real/processed
"""

import argparse
import json
import logging
import os

from predicators.approaches import ApproachFailure, ApproachTimeout
from unipred.deploy import Deployment, load_image, parse_atoms, \
    parse_objects


def main() -> None:
    """Plan for the scene given on the command line (see module doc)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", help="a session folder (raw format)")
    parser.add_argument("--image")
    parser.add_argument("--objects", help="name:type,name:type,...")
    parser.add_argument("--goal", help='"Name(a, b) Name(c) ..."')
    parser.add_argument("--robot_config", default="configs/real_robot.yaml")
    parser.add_argument("--detector", help="override the config's detector")
    parser.add_argument("--encoder", help="override the config's encoder")
    args, training_flags = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.session:
        with open(os.path.join(args.session, "session.json"),
                  encoding="utf-8") as f:
            session = json.load(f)
        image_file = os.path.join(args.session, "0.png")
        objects, goal_atoms = session["objects"], session["goal"]
    else:
        if not (args.image and args.objects and args.goal):
            parser.error("give --session, or --image, --objects and --goal")
        image_file = args.image
        objects, goal_atoms = parse_objects(args.objects), parse_atoms(
            args.goal)
    deployment = Deployment(training_flags, args.robot_config, args.detector,
                            args.encoder)
    state, observation = deployment.observe(load_image(image_file), objects,
                                            image_file=image_file)
    for name in sorted(observation.boxes):
        box = ", ".join(f"{v:.0f}" for v in observation.boxes[name])
        logging.info(f"{name}: box ({box})")
    goal = deployment.goal(goal_atoms, state)
    try:
        plan = deployment.plan(state, goal)
    except (ApproachFailure, ApproachTimeout) as err:
        logging.info(f"No plan: {err}")
        return
    logging.info(f"Plan ({len(plan)} steps):")
    for step, nsrt in enumerate(plan):
        objects_str = ", ".join(o.name for o in nsrt.option_objs)
        logging.info(f"  {step + 1}. {nsrt.option.name}({objects_str})")


if __name__ == "__main__":
    main()
