"""Ground-truth options of Table Clean, shared by the simulated and the real
domain (the real robot runs its own skills with the same names)."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type
from unipred.envs.table_clean_sim import PICK, PLACE, PULL_IN, PUSH_OUT, \
    WIPE, TableCleanSimEnv

_ENV = TableCleanSimEnv


def _act(primitive: int, x: float, y: float) -> Action:
    return Action(np.array([primitive, x, y], dtype=np.float32))


def _at(primitive: int, index: int):
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
    offset = 0.7 * _ENV.box_half_size * np.clip(params, -1, 1)
    return _act(PLACE,
                state.get(box, "x") + offset[0],
                state.get(box, "y") + offset[1])


def _place_on_table(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
    """Place the held object on the free part of the table; params in
    [0, 1]^2 choose the spot."""
    del state, memory, objects  # unused
    (x0, x1), (y0, y1) = _ENV.free_x, _ENV.free_y
    u, v = np.clip(params, 0, 1)
    return _act(PLACE, x0 + u * (x1 - x0), y0 + v * (y1 - y0))


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
        spot_space = Box(0, 1, (2, ))
        return {
            utils.SingletonParameterizedOption(
                "PickToyFromTable", _at(PICK, 1), types=[robot, toy, table]),
            utils.SingletonParameterizedOption("PlaceToyInBox",
                                               _place_in_box,
                                               types=[robot, toy, box],
                                               params_space=offset_space),
            utils.SingletonParameterizedOption(
                "PickWiperFromBox", _at(PICK, 1),
                types=[robot, wiper, box, table]),
            utils.SingletonParameterizedOption(
                "PickWiperFromTable", _at(PICK, 1),
                types=[robot, wiper, table]),
            utils.SingletonParameterizedOption("PlaceWiperOnTable",
                                               _place_on_table,
                                               types=[robot, wiper, table],
                                               params_space=spot_space),
            utils.SingletonParameterizedOption(
                "PlaceWiperInBox", _place_in_box,
                types=[robot, wiper, box, table], params_space=offset_space),
            utils.SingletonParameterizedOption(
                "PushBoxOut", _at(PUSH_OUT, 1), types=[robot, box, table]),
            utils.SingletonParameterizedOption(
                "PullBoxIn", _at(PULL_IN, 1), types=[robot, box, table]),
            utils.SingletonParameterizedOption(
                "WipeTable", _at(WIPE, 2), types=[robot, wiper, table, box]),
        }
