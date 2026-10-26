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

import itertools
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Set, Tuple

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
        by_type: Dict[str, List[Object]] = {}
        for obj in sorted(objects):
            by_type.setdefault(obj.type.name, []).append(obj)
        true_args = {
            tuple(a.objects)
            for a in atoms if a.predicate == self.base
        }
        domains = [by_type.get(t.name, []) for t in self.base.types]
        free_positions = [
            i for i in range(self.base.arity) if i not in self.quantified
        ]
        values: Dict[Tuple[Object, ...], List[bool]] = {}
        for args in itertools.product(*domains):
            if len(set(args)) < len(args):
                continue
            free = tuple(args[i] for i in free_positions)
            values.setdefault(free, []).append(
                (args in true_args) != self.negated)
        head_atoms = set()
        for free, vals in values.items():
            if self.quantifier == "Exists":
                truth = any(vals)
            else:  # ForAll, or a negated basic atom (a single value)
                truth = all(vals)
            if truth:
                head_atoms.add(GroundAtom(self.head, list(free)))
        return head_atoms
