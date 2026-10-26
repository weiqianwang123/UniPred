"""Ground-truth options of Packing (see unipred/envs/packing.py)."""

from typing import Dict, Sequence, Set

import numpy as np
from gymnasium.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type
from unipred.envs.packing import PICK, PLACE, SHIFT, PackingEnv

# A place parameter u in [-1, 1] puts the item at u * PLACE_RANGE from the
# region's anchor (its center while the divider is centered).
PLACE_RANGE = PackingEnv.box_width / 4


def _act(primitive: int, x: float, y: float, fraction: float = 0.0) -> Action:
    return Action(
        np.clip(np.array([primitive, x, y, fraction], dtype=np.float32), 0,
                [SHIFT, 1, 1, 1]).astype(np.float32))


def _pick(state: State, memory: Dict, objects: Sequence[Object],
          params: Array) -> Action:
    del memory, params  # unused
    item = objects[1]
    return _act(PICK, state.get(item, "x"), state.get(item, "y"))


def _place(state: State, memory: Dict, objects: Sequence[Object],
           params: Array) -> Action:
    del memory  # unused
    region = objects[2]
    x = state.get(region, "x") + float(params[0]) * PLACE_RANGE
    return _act(PLACE, x, state.get(region, "y"))


def _shift(state: State, memory: Dict, objects: Sequence[Object],
           params: Array) -> Action:
    """Shift the divider between the two regions (at the box center) so
    that objects[1] becomes wider by the fraction params[0] of the
    maximum shift."""
    del memory  # unused
    _, region, other = objects
    x_region, x_other = state.get(region, "x"), state.get(other, "x")
    toward_other = 1.0 if x_region < x_other else -1.0
    fraction = 0.5 + toward_other * PackingEnv.max_shift * float(params[0])
    return _act(SHIFT, (x_region + x_other) / 2, state.get(region, "y"),
                fraction)


class PackingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for Packing."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"packing"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        robot, item, region = types["robot"], types["item"], types["region"]
        option = utils.SingletonParameterizedOption
        return {
            option("Pick", _pick, types=[robot, item]),
            option("PlaceSmall",
                   _place,
                   types=[robot, item, region],
                   params_space=Box(-1, 1, (1, ))),
            option("PlaceLarge",
                   _place,
                   types=[robot, item, region],
                   params_space=Box(-1, 1, (1, ))),
            option("Shift",
                   _shift,
                   types=[robot, region, region],
                   params_space=Box(0, 1, (1, ))),
        }
