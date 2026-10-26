"""The derived closure used by task planning (see predicators/hooks.py)."""

from __future__ import annotations

from typing import Sequence, Set

from predicators.hooks import AtomsClosure
from predicators.structs import GroundAtom, Object
from unipred.derived.rules import DerivedRule


def make_closure(rules: Sequence[DerivedRule]) -> AtomsClosure:
    """Return a closure that recomputes the heads of the rules.

    All atoms of derived predicates are removed first and then recomputed
    from the remaining (basic) atoms, so that derived atoms that stop
    holding after an operator are not carried over.
    """
    rules = list(rules)
    heads = frozenset(rule.head for rule in rules)
    assert not heads & {rule.base for rule in rules}, \
        "Derived predicates must be defined from basic predicates."

    def _closure(atoms: Set[GroundAtom],
                 objects: Set[Object]) -> Set[GroundAtom]:
        basic = {atom for atom in atoms if atom.predicate not in heads}
        derived: Set[GroundAtom] = set()
        for rule in rules:
            derived |= rule.evaluate(basic, objects)
        return basic | derived

    return AtomsClosure(heads, _closure)
