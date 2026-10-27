"""Ground-truth NSRTs of Cluttered Retrieval, shared by the simulated and
the real domain.

PickTarget needs Clear(target), which no operator adds: it is the derived
predicate ForAll ?obstacle. Not Blocks(?obstacle, ?target), computed by the
atoms closure of get_atoms_closure() while planning. Clearing an obstacle
deletes Blocks(obstacle, target) for the target it names; an obstacle
blocks at most one target.
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


def _place_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator,
                   objects: Sequence[Object]) -> Array:
    """A spot inside the box (see _place_in_box in options.py)."""
    del state, goal, objects  # unused
    return rng.uniform(-1, 1, size=2).astype(np.float32)


class ClutteredRetrievalGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for Cluttered Retrieval."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cluttered_retrieval_sim", "cluttered_retrieval_real"}

    @classmethod
    def get_atoms_closure(
            cls, env_name: str,
            predicates: Dict[str, Predicate]) -> Optional[AtomsClosure]:
        if "Clear" not in predicates or "Blocks" not in predicates:
            return None
        return make_closure([
            DerivedRule(head=predicates["Clear"],
                        base=predicates["Blocks"],
                        negated=True,
                        quantifier="ForAll",
                        quantified=(0, ))
        ])

    @classmethod
    def get_nsrts(cls, env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        robot = Variable("?robot", types["robot"])
        target = Variable("?target", types["target"])
        obstacle = Variable("?obstacle", types["obstacle"])
        box = Variable("?box", types["box"])

        def _atoms(specs: Sequence[Tuple]) -> Set[LiftedAtom]:
            return {
                LiftedAtom(predicates[name], list(args))
                for name, *args in specs
            }

        def _nsrt(name: str,
                  parameters: Sequence[Variable],
                  pre: Sequence[Tuple],
                  add: Sequence[Tuple],
                  delete: Sequence[Tuple] = (),
                  sampler=null_sampler) -> NSRT:
            return NSRT(name, list(parameters), _atoms(pre), _atoms(add),
                        _atoms(delete), set(), options[name],
                        list(parameters), sampler)

        hand_empty = ("HandEmpty", robot)
        holding = ("Holding", robot, target)
        blocks = ("Blocks", obstacle, target)
        return {
            _nsrt("PickTarget", [robot, target],
                  pre=[hand_empty, ("OnTable", target), ("Clear", target)],
                  add=[holding],
                  delete=[hand_empty, ("OnTable", target)]),
            _nsrt("PlaceTargetInBox", [robot, target, box],
                  pre=[holding],
                  add=[("InBox", target, box), hand_empty],
                  delete=[holding],
                  sampler=_place_sampler),
            _nsrt("RemoveObstacle", [robot, obstacle, target],
                  pre=[hand_empty, ("Pickable", obstacle), blocks],
                  add=[],
                  delete=[blocks]),
            _nsrt("PushObstacle", [robot, obstacle, target],
                  pre=[hand_empty, ("Bulky", obstacle), blocks],
                  add=[],
                  delete=[blocks]),
        }
