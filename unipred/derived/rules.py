"""Derived predicates: negated and quantified forms of a basic predicate.

A derived predicate is defined from a learned basic predicate B(x0, ...,
xn) by an optional negation and an optional quantifier over some argument
positions, e.g. ForAll ?toy. Not B(?toy, ?table). Its free arguments are
the unquantified ones, in order.

The semantics mirror the classifiers that IVNTR builds for these forms
(BilevelLearningApproach.generate_classifier), so that the atoms computed
from basic atoms by the closure agree with the derived predicate's own
classifier: the negation applies inside the quantifier, groundings that
repeat an object are skipped, objects are matched by their exact type, and
a quantifier over an empty set of groundings is false.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Collection, Dict, FrozenSet, List, Optional, Set, Tuple

from predicators.structs import GroundAtom, Object, Predicate


@dataclass(frozen=True)
class DerivedRule:
    """head(free args) := Q x_quantified. [not] base(x0, ..., xn)."""
    head: Predicate
    base: Predicate
    negated: bool = False
    quantifier: Optional[str] = None  # None, "ForAll" or "Exists"
    quantified: Tuple[int, ...] = ()  # quantified argument positions of base

    def __post_init__(self) -> None:
        assert self.quantifier in (None, "ForAll", "Exists")
        assert (self.quantifier is None) == (not self.quantified)
        assert self.negated or self.quantifier is not None, \
            "A rule without negation or quantifier is the base itself."
        free_types = [
            t for i, t in enumerate(self.base.types)
            if i not in self.quantified
        ]
        assert list(self.head.types) == free_types, \
            f"{self.head.name} must take the free arguments of " \
            f"{self.base.name}."

    def evaluate(self, atoms: Set[GroundAtom],
                 objects: Collection[Object]) -> Set[GroundAtom]:
        """The true head atoms, given the true atoms of the base."""
        true_args = {
            tuple(a.objects)
            for a in atoms if a.predicate == self.base
        }
        head_atoms = set()
        for head_atom, groundings in _groundings(self, frozenset(objects)):
            values = ((args in true_args) != self.negated
                      for args in groundings)
            if any(values) if self.quantifier == "Exists" else all(values):
                head_atoms.add(head_atom)
        return head_atoms


@functools.lru_cache(maxsize=4096)
def _groundings(
    rule: DerivedRule, objects: FrozenSet[Object]
) -> List[Tuple[GroundAtom, List[Tuple[Object, ...]]]]:
    """Each head atom of the rule over the objects, with the groundings of
    the base that decide it (only heads with at least one grounding)."""
    by_type: Dict[str, List[Object]] = {}
    for obj in sorted(objects):
        by_type.setdefault(obj.type.name, []).append(obj)
    domains = [by_type.get(t.name, []) for t in rule.base.types]
    free_positions = [
        i for i in range(rule.base.arity) if i not in rule.quantified
    ]
    groundings: Dict[Tuple[Object, ...], List[Tuple[Object, ...]]] = {}
    for args in itertools.product(*domains):
        if len(set(args)) < len(args):
            continue
        free = tuple(args[i] for i in free_positions)
        groundings.setdefault(free, []).append(args)
    return [(GroundAtom(rule.head, list(free)), args_list)
            for free, args_list in groundings.items()]
