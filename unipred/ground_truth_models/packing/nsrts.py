"""Ground-truth NSRTs of Packing.

Placing needs Free(region), the derived predicate ForAll ?item. Not
In(?item, ?region), which no operator adds; the atoms closure of
get_atoms_closure() computes it while planning. The samplers read the
simulator's scene (the divider positions and item sizes), which only the
ground truth may.
"""

from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.hooks import AtomsClosure
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler
from unipred.derived.closure import make_closure
from unipred.derived.rules import DerivedRule
from unipred.envs.packing import PackingEnv
from unipred.ground_truth_models.packing.options import PLACE_RANGE

# Shift parameters that leave the wide region wide enough for a large item
# and the narrow one for a small item (see PackingEnv.max_shift).
SHIFT_PARAMS = (0.15, 0.3)


def _place_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator,
                   objects: Sequence[Object]) -> Array:
    """A spot where the held item fits in the region's current bounds."""
    del goal  # unused
    _, item, region = objects
    scene = state.simulator_state
    left, right = PackingEnv.region_bounds(scene, region.name)
    half = PackingEnv.item_sizes[scene.item_sizes[item.name]] / 2
    x = rng.uniform(left + half, right - half)
    return np.array([(x - state.get(region, "x")) / PLACE_RANGE],
                    dtype=np.float32)


def _shift_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator,
                   objects: Sequence[Object]) -> Array:
    del state, goal, objects  # unused
    return np.array([rng.uniform(*SHIFT_PARAMS)], dtype=np.float32)


class PackingGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for Packing."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"packing"}

    @classmethod
    def get_atoms_closure(
            cls, env_name: str,
            predicates: Dict[str, Predicate]) -> Optional[AtomsClosure]:
        if "Free" not in predicates or "In" not in predicates:
            return None
        return make_closure([
            DerivedRule(head=predicates["Free"],
                        base=predicates["In"],
                        negated=True,
                        quantifier="ForAll",
                        quantified=(0, ))
        ])

    @classmethod
    def get_nsrts(cls, env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        robot = Variable("?robot", types["robot"])
        item = Variable("?item", types["item"])
        region = Variable("?region", types["region"])
        other = Variable("?other", types["region"])

        def _atoms(specs: Sequence[Tuple]) -> Set[LiftedAtom]:
            return {
                LiftedAtom(predicates[name], list(args))
                for name, *args in specs
            }

        def _nsrt(name: str,
                  option: str,
                  parameters: Sequence[Variable],
                  pre: Sequence[Tuple],
                  add: Sequence[Tuple],
                  delete: Sequence[Tuple] = (),
                  sampler=null_sampler) -> NSRT:
            return NSRT(name, list(parameters), _atoms(pre), _atoms(add),
                        _atoms(delete), set(), options[option],
                        list(parameters), sampler)

        place_effects = {
            "add": [("In", item, region), ("Packed", item),
                    ("HandEmpty", robot)],
            "delete": [("Holding", robot, item)],
        }
        return {
            _nsrt("Pick",
                  "Pick", [robot, item],
                  pre=[("HandEmpty", robot), ("OnTable", item)],
                  add=[("Holding", robot, item)],
                  delete=[("HandEmpty", robot), ("OnTable", item)]),
            _nsrt("PlaceSmall",
                  "PlaceSmall", [robot, item, region],
                  pre=[("Holding", robot, item), ("IsSmall", item),
                       ("Free", region)],
                  sampler=_place_sampler,
                  **place_effects),
            _nsrt("PlaceLarge",
                  "PlaceLarge", [robot, item, region],
                  pre=[("Holding", robot, item), ("IsLarge", item),
                       ("Free", region), ("Wide", region)],
                  sampler=_place_sampler,
                  **place_effects),
            _nsrt("Shift",
                  "Shift", [robot, region, other],
                  pre=[("SameBox", region, other), ("Free", region),
                       ("Free", other), ("Centered", region),
                       ("Centered", other)],
                  add=[("Wide", region)],
                  delete=[("Centered", region), ("Centered", other)],
                  sampler=_shift_sampler),
        }
