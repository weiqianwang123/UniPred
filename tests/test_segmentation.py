"""Tests for segment helpers used by the operator belief score function."""

import numpy as np

from predicators.nsrt_learning.segmentation import without_effects_on
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Predicate, Segment, State, Type

_BLOCK = Type("block", ["x"])
_ON = Predicate("On", [_BLOCK], lambda s, o: True)
_OFF = Predicate("Off", [_BLOCK], lambda s, o: True)
_DERIVED = Predicate("NoneOn", [], lambda s, o: True)


def test_without_effects_on():
    """Atoms of the given predicates keep their initial values, so only the
    other predicates are effects; the original segment is unchanged."""
    block = _BLOCK("b0")
    state = State({block: np.array([0.0])})
    traj = LowLevelTrajectory([state, state],
                              [Action(np.zeros(1, dtype=np.float32))])
    init_atoms = {GroundAtom(_OFF, [block]), GroundAtom(_DERIVED, [])}
    final_atoms = {GroundAtom(_ON, [block])}
    segment = Segment(traj, init_atoms, final_atoms)
    goal = {GroundAtom(_ON, [block])}
    segment.set_goal(goal)
    frozen = without_effects_on(segment, {_DERIVED})
    assert frozen.add_effects == {GroundAtom(_ON, [block])}
    assert frozen.delete_effects == {GroundAtom(_OFF, [block])}
    assert frozen.init_atoms == init_atoms
    assert frozen.get_goal() == goal
    assert segment.delete_effects == init_atoms
