"""Ground-truth NSRTs of Table Clean, shared by the simulated and the real
domain.

WipeTable needs NoToyOnTable(table), which no operator adds: it is the
derived predicate ForAll ?toy. Not ToyOnTable(?toy, ?table), computed by
the atoms closure of get_atoms_closure() while planning.
"""

from typing import Dict, Optional, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.hooks import AtomsClosure
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler
from unipred.derived.closure import make_closure
from unipred.derived.rules import DerivedRule


def _uniform_sampler(low: float, high: float):

    def _sampler(state, goal, rng: np.random.Generator, objects):
        del state, goal, objects  # unused
        return rng.uniform(low, high, size=2).astype(np.float32)

    return _sampler


class TableCleanGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for Table Clean."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"table_clean_sim", "table_clean_real"}

    @classmethod
    def get_atoms_closure(
            cls, env_name: str,
            predicates: Dict[str, Predicate]) -> Optional[AtomsClosure]:
        if "NoToyOnTable" not in predicates or "ToyOnTable" not in predicates:
            return None
        return make_closure([
            DerivedRule(head=predicates["NoToyOnTable"],
                        base=predicates["ToyOnTable"],
                        negated=True,
                        quantifier="ForAll",
                        quantified=(0, ))
        ])

    @classmethod
    def get_nsrts(cls, env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        robot = Variable("?robot", types["robot"])
        toy = Variable("?toy", types["toy"])
        wiper = Variable("?wiper", types["wiper"])
        box = Variable("?box", types["box"])
        table = Variable("?table", types["table"])

        def _atoms(*specs):
            return {
                LiftedAtom(predicates[name], list(args))
                for name, *args in specs
            }

        def _nsrt(name, parameters, pre, add, delete, sampler=null_sampler):
            return NSRT(name, parameters, _atoms(*pre), _atoms(*add),
                        _atoms(*delete), set(), options[name], parameters,
                        sampler)

        hand_empty = ("HandEmpty", robot)
        holding_toy = ("HoldingToy", robot, toy)
        holding_wiper = ("HoldingWiper", robot, wiper)
        box_near = ("BoxNear", box, table)
        box_far = ("BoxFar", box, table)
        return {
            _nsrt("PickToyFromTable", [robot, toy, table],
                  pre=[hand_empty, ("ToyOnTable", toy, table)],
                  add=[holding_toy],
                  delete=[hand_empty, ("ToyOnTable", toy, table)]),
            _nsrt("PlaceToyInBox", [robot, toy, box],
                  pre=[holding_toy],
                  add=[("ToyInBox", toy, box), hand_empty],
                  delete=[holding_toy],
                  sampler=_uniform_sampler(-1, 1)),
            _nsrt("PickWiperFromBox", [robot, wiper, box, table],
                  pre=[hand_empty, ("WiperInBox", wiper, box), box_near],
                  add=[holding_wiper],
                  delete=[hand_empty, ("WiperInBox", wiper, box)]),
            _nsrt("PickWiperFromTable", [robot, wiper, table],
                  pre=[hand_empty, ("WiperOnTable", wiper, table)],
                  add=[holding_wiper],
                  delete=[hand_empty, ("WiperOnTable", wiper, table)]),
            _nsrt("PlaceWiperOnTable", [robot, wiper, table],
                  pre=[holding_wiper],
                  add=[("WiperOnTable", wiper, table), hand_empty],
                  delete=[holding_wiper],
                  sampler=_uniform_sampler(0, 1)),
            _nsrt("PlaceWiperInBox", [robot, wiper, box, table],
                  pre=[holding_wiper, box_near],
                  add=[("WiperInBox", wiper, box), hand_empty],
                  delete=[holding_wiper],
                  sampler=_uniform_sampler(-1, 1)),
            _nsrt("PushBoxOut", [robot, box, table],
                  pre=[hand_empty, box_near],
                  add=[box_far],
                  delete=[box_near]),
            _nsrt("PullBoxIn", [robot, box, table],
                  pre=[hand_empty, box_far],
                  add=[box_near],
                  delete=[box_far]),
            _nsrt("WipeTable", [robot, wiper, table, box],
                  pre=[holding_wiper, ("NoToyOnTable", table), box_far],
                  add=[("TableClean", table)],
                  delete=[]),
        }
