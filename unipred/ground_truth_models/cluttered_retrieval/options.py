"""Ground-truth options of Cluttered Retrieval, shared by the simulated and
the real domain.

In the simulated domain, each option applies one primitive of
ClutteredRetrievalSimEnv at the pose of one of its object arguments. The
real domain has the same option names and types; a robot executes them
through its own skills (see examples/real_robot), so their policies are not
run there. Clearing an obstacle names the target it blocks, as its effect
is on both.
"""

from typing import Dict, Sequence, Set

import numpy as np
from gymnasium.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type
from unipred.envs.cluttered_retrieval_sim import PICK, PLACE, PUSH, REMOVE, \
    ClutteredRetrievalSimEnv


def _act(primitive: int, x: float, y: float, direction: float = 0.0) -> Action:
    return Action(np.array([primitive, x, y, direction], dtype=np.float32))


def _pick(state: State, memory: Dict, objects: Sequence[Object],
          params: Array) -> Action:
    del memory, params  # unused
    target = objects[1]
    return _act(PICK, state.get(target, "x"), state.get(target, "y"))


def _place_in_box(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> Action:
    """Place the held target in the box (objects[2]); params in [-1, 1]^2
    choose the spot inside the box."""
    del memory  # unused
    box = objects[2]
    offset = 0.7 * ClutteredRetrievalSimEnv.box_half_size * np.clip(
        params, -1, 1)
    return _act(PLACE,
                state.get(box, "x") + offset[0],
                state.get(box, "y") + offset[1])


def _remove(state: State, memory: Dict, objects: Sequence[Object],
            params: Array) -> Action:
    del memory, params  # unused
    obstacle = objects[1]
    return _act(REMOVE, state.get(obstacle, "x"), state.get(obstacle, "y"))


def _push(state: State, memory: Dict, objects: Sequence[Object],
          params: Array) -> Action:
    """Push the obstacle sideways, away from the target it blocks."""
    del memory, params  # unused
    _, obstacle, target = objects
    away = state.get(obstacle, "x") - state.get(target, "x")
    return _act(PUSH, state.get(obstacle, "x"), state.get(obstacle, "y"),
                1.0 if away >= 0 else -1.0)


class ClutteredRetrievalGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for Cluttered Retrieval."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cluttered_retrieval_sim", "cluttered_retrieval_real"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        robot, target = types["robot"], types["target"]
        obstacle, box = types["obstacle"], types["box"]
        option = utils.SingletonParameterizedOption
        return {
            option("PickTarget", _pick, types=[robot, target]),
            option("PlaceTargetInBox",
                   _place_in_box,
                   types=[robot, target, box],
                   params_space=Box(-1, 1, (2, ))),
            option("RemoveObstacle",
                   _remove,
                   types=[robot, obstacle, target]),
            option("PushObstacle", _push, types=[robot, obstacle, target]),
        }
