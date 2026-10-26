"""Extension hooks for packages that build on predicators.

Atoms closure
-------------
Some domains use derived predicates: predicates whose truth value is
defined by other predicates (for example, "no toy is on the table"). No
operator changes them directly, so after applying an operator in task
planning they must be recomputed from the remaining atoms. An extension
installs a function that does this with set_atoms_closure(). The task
planner (A* in planning.py) then calls atoms_closure() on the initial
atoms and on every successor it creates, with all task objects passed in.
The closure receives a copy of the atoms and returns the full atom set,
typically by removing every derived atom and recomputing them from the
basic atoms.

Derived atoms are invisible to delete-relaxation reachability analysis,
so while a closure is installed the planner skips the dr-reachability
filter on ground NSRTs and the dr-reachability goal check. Fast Downward
cannot call the closure, so planning with "fdopt" or "fdsat" raises an
error while a closure is installed, and delete-relaxation heuristics would
treat derived preconditions as unreachable; check_closure_settings() checks
both before a long run.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Collection, Iterator, Optional, Set

from predicators.settings import CFG
from predicators.structs import GroundAtom, Object

AtomsClosure = Callable[[Set[GroundAtom], Set[Object]], Set[GroundAtom]]

# Derived atoms are never operator effects, so delete-relaxation heuristics
# treat derived preconditions as unreachable.
DELETE_RELAXATION_HEURISTICS = ("hadd", "hmax", "hff", "hsa", "lmcut")

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
    if CFG.sesame_task_planning_heuristic in DELETE_RELAXATION_HEURISTICS:
        raise ValueError(
            "Planning with derived predicates needs a heuristic that sees "
            "derived atoms, such as --sesame_task_planning_heuristic "
            "goal_count.")
