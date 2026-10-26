"""Extension hooks for packages that build on predicators.

Atoms closure
-------------
Some domains use derived predicates: predicates whose truth value is
defined by other predicates (for example, "no toy is on the table"). No
operator changes them directly, so after applying an operator in task
planning they must be recomputed from the remaining atoms. An extension
installs an AtomsClosure that does this with set_atoms_closure() (or
temporary_atoms_closure()). The task planner (A* in planning.py) then
calls atoms_closure() on the initial atoms and on every successor it
creates, with all task objects passed in. The closure receives a copy of
the atoms and returns the full atom set, typically by removing every
derived atom and recomputing them from the basic atoms.

Derived atoms are never operator effects, so the delete relaxation (dr-
reachability and the delete-relaxation heuristics) cannot tell when they
become true. While a closure is installed, both treat every atom of a
derived predicate as reachable: optimistic, so no operator or plan is
ruled out wrongly. Fast Downward cannot call the closure, so planning with
"fdopt" or "fdsat" raises an error while a closure is installed;
check_closure_settings() checks this before a long run.
"""

from __future__ import annotations

import contextlib
import itertools
from dataclasses import dataclass
from typing import Callable, Collection, FrozenSet, Iterator, Optional, Set

from predicators.settings import CFG
from predicators.structs import GroundAtom, Object, Predicate


@dataclass(frozen=True)
class AtomsClosure:
    """Recomputes the atoms of the derived predicates from the other
    atoms: function(atoms, objects) returns the full atom set."""
    derived_predicates: FrozenSet[Predicate]
    function: Callable[[Set[GroundAtom], Set[Object]], Set[GroundAtom]]

    def __call__(self, atoms: Set[GroundAtom],
                 objects: Set[Object]) -> Set[GroundAtom]:
        return self.function(atoms, objects)


_ATOMS_CLOSURE: Optional[AtomsClosure] = None


def set_atoms_closure(closure: Optional[AtomsClosure]) -> None:
    """Install the atoms closure used by task planning, or remove it with
    None."""
    global _ATOMS_CLOSURE  # pylint: disable=global-statement
    _ATOMS_CLOSURE = closure


def closure_active() -> bool:
    """Whether an atoms closure is installed."""
    return _ATOMS_CLOSURE is not None


def atoms_closure(atoms: Set[GroundAtom],
                  objects: Collection[Object]) -> Set[GroundAtom]:
    """Apply the installed closure to the atoms.

    Without a closure, the atoms are returned unchanged (same object).
    """
    if _ATOMS_CLOSURE is None:
        return atoms
    return _ATOMS_CLOSURE(set(atoms), set(objects))


def all_derived_atoms(objects: Collection[Object]) -> Set[GroundAtom]:
    """Every ground atom of the installed closure's derived predicates over
    the objects (none without a closure): what the delete relaxation
    assumes to be reachable."""
    if _ATOMS_CLOSURE is None:
        return set()
    atoms = set()
    for pred in _ATOMS_CLOSURE.derived_predicates:
        domains = [[o for o in objects if o.is_instance(t)]
                   for t in pred.types]
        for args in itertools.product(*domains):
            atoms.add(GroundAtom(pred, list(args)))
    return atoms


@contextlib.contextmanager
def temporary_atoms_closure(
        closure: Optional[AtomsClosure]) -> Iterator[None]:
    """Install a closure for the duration of a with-block, then restore the
    previously installed one."""
    previous = _ATOMS_CLOSURE
    set_atoms_closure(closure)
    try:
        yield
    finally:
        set_atoms_closure(previous)


def check_closure_settings() -> None:
    """Raise if the planner settings cannot plan with an atoms closure."""
    if CFG.sesame_task_planner != "astar":
        raise ValueError("Planning with derived predicates needs "
                         "--sesame_task_planner astar.")
