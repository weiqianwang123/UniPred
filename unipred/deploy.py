"""Deploying a model learned on a real domain: from images to plans.

A Deployment takes the flags of the training run (the same as for
scripts/run.sh, e.g. --config configs/table_clean_real/unipred.yaml --seed
0 --approach_dir outputs/...), loads the learned approach and sets up the
perception of a real-robot config (configs/real_robot.yaml). For an image
of a scene it computes the objects' features with the encoder the training
data was processed with; the learned classifiers (and, for the given
predicates, classifiers trained on the annotations; see
unipred/envs/offline_real.py) ground the atoms, and the learned operators
give a plan. examples/real_robot/plan_from_image.py and
closed_loop_example.py use it.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task, _GroundNSRT
from unipred.config import expand_config_args
from unipred.envs.offline_real import OfflineRealEnv
from unipred.perception.detector import Box
from unipred.perception.scene import SceneObservation, ScenePerceiver


class Deployment:
    """A learned approach on a real domain, with its perception (see module
    doc)."""

    def __init__(self,
                 flags: Sequence[str],
                 robot_config: str,
                 detector: Optional[str] = None,
                 encoder: Optional[str] = None) -> None:
        argv = sys.argv
        sys.argv = [argv[0]] + expand_config_args(list(flags))
        try:
            utils.update_config(utils.parse_args())
        finally:
            sys.argv = argv
        torch.set_num_threads(CFG.torch_num_threads)
        env = create_new_env(CFG.env, do_cache=True, use_gui=False)
        if not isinstance(env, OfflineRealEnv):
            raise ValueError(f"{CFG.env} is not a real-robot domain.")
        self.env = env
        predicates, _ = utils.parse_config_excluded_predicates(env)
        train_tasks = [
            utils.strip_task(t.task, predicates)
            for t in env.get_train_tasks()
        ]
        self.approach = create_approach(CFG.approach, predicates,
                                        get_gt_options(env.get_name()),
                                        env.types, env.action_space,
                                        train_tasks)
        if not isinstance(self.approach, BilevelPlanningApproach):
            raise ValueError("Deployment needs an approach that plans with "
                             f"operators, not {CFG.approach}.")
        if self.approach.is_learning_based:
            self.approach.load()
        self.perceiver = ScenePerceiver.from_config(robot_config, detector,
                                                    encoder)
        if self.perceiver.encoder.backend != env.encoder:
            raise ValueError(
                f"The training data was encoded with {env.encoder}, but the "
                f"perception uses {self.perceiver.encoder.backend}.")

    def observe(
        self,
        image: np.ndarray,
        objects: Mapping[str, str],
        previous: Optional[Mapping[str, Box]] = None,
        image_file: str = ""
    ) -> Tuple[State, SceneObservation]:
        """The state of the objects (name -> type) in an RGB image, and what
        perception found (its boxes are the previous ones for the next
        image). The image file, if any, is kept in the state."""
        observation = self.perceiver.perceive(image, objects, previous)
        state = self.env.scene_state(dict(objects), observation.features,
                                     image_file)
        return state, observation

    def goal(self, atoms: Sequence[Tuple[str, Sequence[str]]],
             state: State) -> Set[GroundAtom]:
        """Goal atoms from (predicate name, object names)."""
        predicates = {p.name: p for p in self.env.predicates}
        objects = {o.name: o for o in state}
        return {
            GroundAtom(predicates[name], [objects[a] for a in args])
            for name, args in atoms
        }

    def plan(self, state: State, goal: Set[GroundAtom]) -> List[_GroundNSRT]:
        """A plan of learned operators from the state to the goal; raises
        ApproachFailure or ApproachTimeout if none is found."""
        self.approach.solve(Task(state, goal), timeout=CFG.timeout)
        return self.approach.get_last_nsrt_plan()


def load_image(path: str) -> np.ndarray:
    """An image file as an RGB array."""
    return np.asarray(Image.open(path).convert("RGB"))


def parse_atoms(text: str) -> List[Tuple[str, List[str]]]:
    """Atoms written as "Name(a, b) Name(c)"."""
    atoms = []
    for chunk in text.replace(")", ")\n").splitlines():
        chunk = chunk.strip().strip(",").strip()
        if not chunk:
            continue
        name, _, args = chunk.partition("(")
        atoms.append((name.strip(), [
            a.strip() for a in args.rstrip(")").split(",") if a.strip()
        ]))
    return atoms


def parse_objects(text: str) -> Dict[str, str]:
    """Objects written as "name:type,name:type"."""
    objects = {}
    for item in text.split(","):
        name, _, object_type = item.strip().partition(":")
        objects[name.strip()] = object_type.strip()
    return objects
