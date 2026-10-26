"""Tests for the simulated environments: task generation, the rules of
Table Clean Sim, and the oracle on small tasks of every domain (Packing in
test_packing.py)."""

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action
from unipred.envs.table_clean_sim import PICK, PUSH_OUT, WIPE, \
    TableCleanSimEnv


def _env(name, **settings):
    utils.reset_config({
        "env": name,
        "num_train_tasks": 2,
        "num_test_tasks": 2,
        "sesame_task_planner": "astar",
        **settings
    })
    return create_new_env(name)


@pytest.mark.parametrize("name", ["satellites", "blocks_onclear",
                                  "table_clean_sim"])
@pytest.mark.parametrize("in_domain_test", [False, True])
def test_oracle_solves_tasks(name, in_domain_test):
    """The ground-truth operators solve train and test tasks."""
    env = _env(name, in_domain_test=in_domain_test)
    approach = OracleApproach(env.predicates, get_gt_options(name),
                              env.types, env.action_space,
                              [t.task for t in env.get_train_tasks()])
    for env_task in env.get_train_tasks() + env.get_test_tasks():
        task = env_task.task
        policy = approach.solve(task, timeout=30)
        state = task.init
        for _ in range(500):
            if task.goal_holds(state):
                break
            state = env.simulate(state, policy(state))
        assert task.goal_holds(state)


def test_table_clean_sim_tasks():
    """Train tasks: two toys on the table, the wiper in the box near the
    robot, the hand empty. Test tasks: three toys and varied starts."""
    env = _env("table_clean_sim", num_test_tasks=20)
    assert isinstance(env, TableCleanSimEnv)
    for env_task in env.get_train_tasks():
        state = env_task.task.init
        toys = state.get_objects(env.toy_type)
        assert len(toys) == 2
        assert all(state.get(t, "on_table") == 1.0 for t in toys)
        box = state.get_objects(env.box_type)[0]
        assert state.get(box, "side") == 0.0
        robot = state.get_objects(env.robot_type)[0]
        assert state.get(robot, "gripper") == 0.0
    starts = set()
    for env_task in env.get_test_tasks():
        state = env_task.task.init
        assert len(state.get_objects(env.toy_type)) == 3
        robot = state.get_objects(env.robot_type)[0]
        box = state.get_objects(env.box_type)[0]
        starts.add((state.get(robot, "gripper"), state.get(box, "side")))
    assert len(starts) > 1


def test_table_clean_sim_constraints():
    """Wiping needs the wiper in hand, no toy on the table and the box on
    the far side; the wiper leaves the box only while the box is near."""
    env = _env("table_clean_sim")
    state = env.get_train_tasks()[0].task.init
    table = state.get_objects(env.table_type)[0]
    box = state.get_objects(env.box_type)[0]
    wiper = state.get_objects(env.wiper_type)[0]

    def _act(s, primitive, x, y):
        return env.simulate(
            s, Action(np.array([primitive, x, y], dtype=np.float32)))

    def _xy(s, obj):
        return s.get(obj, "x"), s.get(obj, "y")

    # Push the box out first: the wiper cannot be taken out any more.
    far = _act(state, PUSH_OUT, *_xy(state, box))
    assert far.get(box, "side") == 1.0
    assert _act(far, PICK, *_xy(far, wiper)).get(wiper, "held") == 0.0
    # With the box near, the wiper can be taken; wiping is refused while
    # the box is near and while toys are on the table.
    holding = _act(state, PICK, *_xy(state, wiper))
    assert holding.get(wiper, "held") == 1.0
    assert _act(holding, WIPE, 0.5, 0.5).get(table, "is_clean") == 0.0
    pushed = _act(holding, PUSH_OUT, *_xy(holding, box))
    assert pushed.get(box, "side") == 1.0
    assert _act(pushed, WIPE, 0.5, 0.5).get(table, "is_clean") == 0.0
    # Without toys on the table, wiping cleans it.
    cleared = pushed.copy()
    for toy in cleared.get_objects(env.toy_type):
        cleared.set(toy, "on_table", 0.0)
        cleared.set(toy, "in_box", 1.0)
    assert _act(cleared, WIPE, 0.5, 0.5).get(table, "is_clean") == 1.0
