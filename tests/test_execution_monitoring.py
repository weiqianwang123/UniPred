"""Tests for the expected atoms execution monitor on plans found with
simulation."""

import numpy as np

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs import create_new_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.structs import GroundAtom, Predicate, State, Task, Type

_BLOCK = Type("block", ["in_box"])
_IN_BOX = Predicate("InBox", [_BLOCK],
                    lambda s, o: s.get(o[0], "in_box") > 0.5)


class _FakeOption:
    """Stands in for a running option."""

    def __init__(self, terminal: bool) -> None:
        self._terminal = terminal

    def terminal(self, state: State) -> bool:
        """Whether the option is done."""
        del state  # unused
        return self._terminal


def test_monitor_checks_atoms_when_option_terminates():
    """Expected atoms of a running option are only checked once it
    terminates."""
    block = _BLOCK("b0")
    outside = State({block: np.array([0.0])})
    inside = State({block: np.array([1.0])})
    expected = {GroundAtom(_IN_BOX, [block])}
    monitor = create_execution_monitor("expected_atoms")
    monitor.reset(Task(outside, expected))
    monitor.update_approach_info([(_FakeOption(terminal=False), expected)])
    assert not monitor.step(outside)
    monitor.update_approach_info([(_FakeOption(terminal=True), expected)])
    assert not monitor.step(inside)
    assert monitor.step(outside)
    # Plans found without simulation give the atoms of the next state.
    monitor.update_approach_info([expected])
    assert monitor.step(outside)


def test_oracle_sim_plan_is_monitored():
    """An oracle plan found with simulation reports the atoms expected after
    each option and runs to the goal without replanning."""
    utils.reset_config({
        "env": "blocks",
        "approach": "oracle",
        "execution_monitor": "expected_atoms",
        "sesame_task_planner": "astar",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "blocks_num_blocks_test": [5],
    })
    env = create_new_env("blocks")
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("oracle", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space, train_tasks)
    task = env.get_test_tasks()[0].task
    policy = approach.solve(task, timeout=10)
    assert approach.get_execution_monitoring_info() == []
    policy(task.init)
    info = approach.get_execution_monitoring_info()
    assert len(info) == 1
    option, expected_atoms = info[0]
    assert option is approach.get_last_plan()[0]
    assert expected_atoms

    monitor = create_execution_monitor("expected_atoms")
    num_replans = 0
    monitor_step = monitor.step

    def _counting_step(state: State) -> bool:
        nonlocal num_replans
        replan = monitor_step(state)
        num_replans += replan
        return replan

    monitor.step = _counting_step  # type: ignore[method-assign]
    cogman = CogMan(approach, create_perceiver("trivial"), monitor)
    cogman.reset(env.get_test_tasks()[0])
    _, solved, _ = run_episode_and_get_observations(cogman,
                                                    env,
                                                    "test",
                                                    0,
                                                    max_num_steps=100,
                                                    do_env_reset=True)
    assert solved
    assert num_replans == 0
