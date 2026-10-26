"""Tests for the atoms closure hook and its planning call sites."""

from typing import Set

import numpy as np
import pytest

from predicators import hooks, utils
from predicators.planning import PlanningFailure, _skeleton_generator, \
    filter_nsrts, run_task_plan_once, skeleton_atoms_sequence, task_plan, \
    task_plan_grounding
from predicators.structs import Action, GroundAtom, LiftedAtom, Object, \
    Predicate, State, STRIPSOperator, Task, Type

_BLOCK = Type("block", ["in_box"])
_IN_BOX = Predicate("InBox", [_BLOCK],
                    lambda s, o: s.get(o[0], "in_box") > 0.5)
_ON_TABLE = Predicate("OnTable", [_BLOCK],
                      lambda s, o: s.get(o[0], "in_box") < 0.5)
# Derived: every block is in the box.
_ALL_IN_BOX = Predicate("AllInBox", [],
                        lambda s, o: all(s.get(b, "in_box") > 0.5 for b in s))
_DONE = Predicate("Done", [], lambda s, o: False)


def _all_in_box_closure(atoms: Set[GroundAtom],
                        objects: Set[Object]) -> Set[GroundAtom]:
    """Recompute AllInBox from the InBox atoms."""
    atoms = {a for a in atoms if a.predicate != _ALL_IN_BOX}
    blocks = [o for o in objects if o.type == _BLOCK]
    if all(GroundAtom(_IN_BOX, [b]) in atoms for b in blocks):
        atoms.add(GroundAtom(_ALL_IN_BOX, []))
    return atoms


def _toy_nsrts():
    """PutInBox(?b) moves a block from the table into the box; Finish()
    needs the derived atom AllInBox()."""
    block_var = _BLOCK("?b")
    put = STRIPSOperator("PutInBox", [block_var],
                         {LiftedAtom(_ON_TABLE, [block_var])},
                         {LiftedAtom(_IN_BOX, [block_var])},
                         {LiftedAtom(_ON_TABLE, [block_var])}, set())
    finish = STRIPSOperator("Finish", [], {LiftedAtom(_ALL_IN_BOX, [])},
                            {LiftedAtom(_DONE, [])}, set(), set())
    policy = lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32))
    put_option = utils.SingletonParameterizedOption("PutInBox",
                                                    policy,
                                                    types=[_BLOCK])
    finish_option = utils.SingletonParameterizedOption("Finish", policy)
    return utils.ops_and_specs_to_dummy_nsrts(
        [put, finish], [(put_option, [block_var]), (finish_option, [])])


def _toy_task(stale_derived_atom: bool = False):
    """Two blocks on the table; the goal is Done()."""
    blocks = [_BLOCK("b0"), _BLOCK("b1")]
    state = State({b: np.array([0.0]) for b in blocks})
    init_atoms = {GroundAtom(_ON_TABLE, [b]) for b in blocks}
    if stale_derived_atom:
        init_atoms.add(GroundAtom(_ALL_IN_BOX, []))
    return state, blocks, init_atoms, {GroundAtom(_DONE, [])}


def _plan(init_atoms, goal, objects):
    """Run A* task planning on the toy domain."""
    ground_nsrts, reachable_atoms = task_plan_grounding(
        init_atoms, set(objects), _toy_nsrts())
    heuristic = utils.create_task_planning_heuristic(
        "goal_count", init_atoms, goal, ground_nsrts,
        {_IN_BOX, _ON_TABLE, _ALL_IN_BOX, _DONE}, objects)
    return next(
        task_plan(init_atoms,
                  goal,
                  ground_nsrts,
                  reachable_atoms,
                  heuristic,
                  seed=0,
                  timeout=10,
                  max_skeletons_optimized=1,
                  objects=objects))


@pytest.fixture(autouse=True)
def _no_closure_after_test():
    yield
    hooks.set_atoms_closure(None)


def test_identity_without_closure():
    """Without a closure, atoms_closure returns its input."""
    utils.reset_config({"env": "blocks"})
    atoms = {GroundAtom(_DONE, [])}
    assert not hooks.closure_active()
    assert hooks.atoms_closure(atoms, set()) is atoms


def test_temporary_closure_restores_previous():
    """temporary_atoms_closure installs a closure only inside the block."""
    hooks.set_atoms_closure(_all_in_box_closure)
    with hooks.temporary_atoms_closure(None):
        assert not hooks.closure_active()
    assert hooks.closure_active()


def test_derived_goal_unreachable_without_closure():
    """Without the closure, AllInBox never becomes true."""
    utils.reset_config({"env": "blocks"})
    _, blocks, init_atoms, goal = _toy_task()
    with pytest.raises(PlanningFailure):
        _plan(init_atoms, goal, blocks)


def test_astar_applies_closure():
    """A* applies the closure to the root (removing stale derived atoms) and
    to every successor."""
    utils.reset_config({"env": "blocks"})
    _, blocks, init_atoms, goal = _toy_task(stale_derived_atom=True)
    hooks.set_atoms_closure(_all_in_box_closure)
    skeleton, atoms_sequence, _ = _plan(init_atoms, goal, blocks)
    assert [n.name for n in skeleton] == ["PutInBox", "PutInBox", "Finish"]
    all_in_box = GroundAtom(_ALL_IN_BOX, [])
    assert [all_in_box in atoms for atoms in atoms_sequence] == \
        [False, False, True, True]


def test_policy_guided_children_apply_closure():
    """Successors generated by an abstract policy also get the closure."""
    utils.reset_config({"env": "blocks"})
    state, blocks, init_atoms, goal = _toy_task()
    hooks.set_atoms_closure(_all_in_box_closure)
    ground_nsrts, _ = task_plan_grounding(init_atoms, set(blocks),
                                          _toy_nsrts())

    def _policy(atoms, objects, policy_goal):
        del objects, policy_goal  # unused
        for ground_nsrt in sorted(ground_nsrts):
            if ground_nsrt.preconditions.issubset(atoms):
                return ground_nsrt
        return None

    metrics = {"num_skeletons_optimized": 0, "num_nodes_created": 0,
               "num_nodes_expanded": 0}
    gen = _skeleton_generator(Task(state, goal),
                              ground_nsrts,
                              init_atoms,
                              lambda atoms: 0.0,
                              seed=0,
                              timeout=10,
                              metrics=metrics,
                              max_skeletons_optimized=1,
                              abstract_policy=_policy,
                              sesame_max_policy_guided_rollout=10)
    skeleton, atoms_sequence = next(gen)
    assert skeleton[-1].name == "Finish"
    assert GroundAtom(_ALL_IN_BOX, []) in atoms_sequence[-2]


def test_closure_skips_reachability_filters():
    """Operators that are only reachable through derived atoms are kept."""
    utils.reset_config({"env": "blocks"})
    state, blocks, init_atoms, goal = _toy_task()
    ground_nsrts, _ = task_plan_grounding(init_atoms, set(blocks),
                                          _toy_nsrts())
    assert "Finish" not in {n.name for n in ground_nsrts}
    hooks.set_atoms_closure(_all_in_box_closure)
    ground_nsrts, _ = task_plan_grounding(init_atoms, set(blocks),
                                          _toy_nsrts())
    assert "Finish" in {n.name for n in ground_nsrts}
    filtered = filter_nsrts(Task(state, goal), init_atoms, ground_nsrts)
    assert "Finish" in {n.name for n in filtered}


def test_run_task_plan_once_with_closure(monkeypatch):
    """Task planning from a state works with A* and is refused by Fast
    Downward, which cannot apply the closure."""
    preds = {_IN_BOX, _ON_TABLE, _ALL_IN_BOX, _DONE}
    state, _, _, goal = _toy_task()
    task = Task(state, goal)
    hooks.set_atoms_closure(_all_in_box_closure)
    utils.reset_config({"env": "blocks", "sesame_task_planner": "astar"})
    plan, _, _ = run_task_plan_once(task, _toy_nsrts(), preds, {_BLOCK},
                                    timeout=10, seed=0,
                                    task_planning_heuristic="goal_count")
    assert [n.name for n in plan] == ["PutInBox", "PutInBox", "Finish"]
    expected = skeleton_atoms_sequence(task, plan, preds)
    assert GroundAtom(_ALL_IN_BOX, []) in expected[2]
    utils.reset_config({"env": "blocks", "sesame_task_planner": "fdopt"})
    monkeypatch.setenv("FD_EXEC_PATH", "unused")
    with pytest.raises(ValueError, match="atoms closure"):
        run_task_plan_once(task, _toy_nsrts(), preds, {_BLOCK},
                           timeout=10, seed=0)


def test_pddl_domain_drops_derived_effects():
    """Derived predicates are not written as operator effects."""
    block_var = _BLOCK("?b")
    bad_op = STRIPSOperator("PutInBox", [block_var],
                            {LiftedAtom(_ON_TABLE, [block_var])}, {
                                LiftedAtom(_IN_BOX, [block_var]),
                                LiftedAtom(_ALL_IN_BOX, [])
                            }, set(), set())
    preds = {_IN_BOX, _ON_TABLE, _ALL_IN_BOX}
    with_derived = utils.create_pddl_domain([bad_op], preds, {_BLOCK}, "toy")
    effects = with_derived.split(":effect")[1]
    assert "AllInBox" in effects
    without = utils.create_pddl_domain([bad_op],
                                       preds, {_BLOCK},
                                       "toy",
                                       derived_predicates={_ALL_IN_BOX})
    assert "AllInBox" not in without.split(":effect")[1]
    assert "(InBox ?b)" in without.split(":effect")[1]
