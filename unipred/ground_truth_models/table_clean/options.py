"""Ground-truth options of Table Clean, shared by the simulated and the real
domain.

In the simulated domain, each option applies one primitive of
TableCleanSimEnv at the pose of one of its object arguments. The real
domain has the same option names and types; a robot executes them through
its own skills (see examples/real_robot), so their policies are not run
there.
"""

from typing import Callable, Dict, Sequence, Set

import numpy as np
from gymnasium.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type
from unipred.envs.table_clean_sim import PICK, PLACE, PULL_IN, PUSH_OUT, \
    WIPE, TableCleanSimEnv

_Policy = Callable[[State, Dict, Sequence[Object], Array], Action]


def _act(primitive: int, x: float, y: float) -> Action:
    return Action(np.array([primitive, x, y], dtype=np.float32))


def _at(primitive: int, index: int) -> _Policy:
    """A policy that applies the primitive at the pose of an argument."""

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory, params  # unused
        obj = objects[index]
        return _act(primitive, state.get(obj, "x"), state.get(obj, "y"))

    return _policy


def _place_in_box(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> Action:
    """Place the held object in the box (objects[2]); params in [-1, 1]^2
    choose the spot inside the box."""
    del memory  # unused
    box = objects[2]
    offset = 0.7 * TableCleanSimEnv.box_half_size * np.clip(params, -1, 1)
    return _act(PLACE,
                state.get(box, "x") + offset[0],
                state.get(box, "y") + offset[1])


class TableCleanGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for Table Clean."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"table_clean_sim", "table_clean_real"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        robot, toy, wiper = types["robot"], types["toy"], types["wiper"]
        box, table = types["box"], types["table"]
        offset_space = Box(-1, 1, (2, ))
        option = utils.SingletonParameterizedOption
        return {
            option("PickToyFromTable",
                   _at(PICK, 1),
                   types=[robot, toy, table]),
            option("PlaceToyInBox",
                   _place_in_box,
                   types=[robot, toy, box],
                   params_space=offset_space),
            option("PickWiperFromBox",
                   _at(PICK, 1),
                   types=[robot, wiper, box]),
            option("PlaceWiperInBox",
                   _place_in_box,
                   types=[robot, wiper, box],
                   params_space=offset_space),
            option("PushBoxOut", _at(PUSH_OUT, 1), types=[robot, box]),
            option("PullBoxIn", _at(PULL_IN, 1), types=[robot, box]),
            option("WipeTable",
                   _at(WIPE, 2),
                   types=[robot, wiper, table, box]),
        }
