"""Tests for the Packing environment and its ground-truth models."""

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import Action
from unipred.envs.packing import PLACE, SHIFT, PackingEnv

_SETTINGS = {
    "env": "packing",
    "perception_encoder": "fake",
    "device": "cpu",
    "num_train_tasks": 2,
    "num_test_tasks": 2,
    "allow_state_allclose_comparison_despite_simulator_state": True,
    "sesame_task_planner": "astar",
    "sesame_use_visited_state_set": True,
}


@pytest.fixture(name="env")
def _env() -> PackingEnv:
    utils.reset_config(_SETTINGS)
    env = create_new_env("packing")
    assert isinstance(env, PackingEnv)
    return env


def test_tasks(env):
    """Train tasks have two boxes and three items, test tasks three boxes
    and four items; crop features are filled in and region widths are not
    features."""
    train, test = env.get_train_tasks()[0], env.get_test_tasks()[0]
    for task, boxes, items in ((train, 2, 3), (test, 3, 4)):
        state = task.init
        assert len(state.get_objects(env.region_type)) == 2 * boxes
        assert len(state.get_objects(env.item_type)) == items
        assert len(task.goal) == items
        for obj in state.get_objects(env.region_type):
            assert np.any(state[obj][2:] != 0)
    assert "width" not in env.region_type.feature_names
    sizes = test.init.simulator_state.item_sizes.values()
    assert sorted(sizes) == ["large", "large", "small", "small"]


def test_shift_then_place_large(env):
    """A large item fits a region only after the divider is shifted, and the
    region's crop shows the change."""
    state = env.get_train_tasks()[0].init
    scene = state.simulator_state
    large = next(i for i in state.get_objects(env.item_type)
                 if scene.item_sizes[i.name] == "large")
    left = next(r for r in state.get_objects(env.region_type)
                if r.name == "box0_left")
    robot = state.get_objects(env.robot_type)[0]
    pick = Action(np.array([0, state.get(large, "x"),
                            state.get(large, "y"), 0], dtype=np.float32))
    holding = env.simulate(state, pick)
    assert holding.get(large, "held") == 1.0
    place_x = env.region_bounds(scene, "box0_left")[0] + 0.078
    place = Action(np.array([PLACE, place_x, env.box_y, 0],
                            dtype=np.float32))
    assert env.simulate(holding, place).get(large, "held") == 1.0
    box_x = env.box_x(0)
    shift = Action(np.array([SHIFT, box_x, env.box_y, 0.6],
                            dtype=np.float32))
    shifted = env.simulate(holding, shift)
    assert not np.allclose(shifted[left][2:], holding[left][2:])
    placed = env.simulate(shifted, place)
    assert placed.get(large, "held") == 0.0
    assert placed.simulator_state.contents["box0_left"] == large.name
    assert placed.get(robot, "gripper") == 0.0


def test_shift_needs_the_regions_of_one_box(env):
    """Shift moves the divider between two regions of one box, and nothing
    for regions of two boxes (halfway between which is a third box)."""
    state = env.get_test_tasks()[0].init
    regions = {r.name: r for r in state.get_objects(env.region_type)}
    robot = state.get_objects(env.robot_type)[0]
    shift = get_gt_options("packing")
    shift = next(o for o in shift if o.name == "Shift")
    for names, moved in ((("box0_left", "box2_right"), None),
                         (("box0_right", "box2_left"), None),
                         (("box1_right", "box1_left"), 1)):
        option = shift.ground([robot] + [regions[n] for n in names],
                              np.array([0.3], dtype=np.float32))
        after = env.simulate(state, option.policy(state))
        dividers = after.simulator_state.dividers
        assert {b for b, f in dividers.items() if f != 0.5} == (
            set() if moved is None else {moved})
    assert dividers[1] < 0.5  # box1_right became wide


def test_oracle_solves_tasks(env):
    """The ground-truth operators solve a train and a test task."""
    options = get_gt_options("packing")
    approach = OracleApproach(env.predicates, options, env.types,
                              env.action_space, env.get_train_tasks())
    for env_task in (env.get_train_tasks()[0], env.get_test_tasks()[0]):
        task = env_task.task
        policy = approach.solve(task, timeout=30)
        state = task.init
        for _ in range(100):
            if task.goal_holds(state):
                break
            state = env.simulate(state, policy(state))
        assert task.goal_holds(state)


def test_place_sampler_without_room(env):
    """Placing a large item into a region that is too narrow still samples
    a parameter in the option's range (policy baselines try this)."""
    state = env.get_train_tasks()[0].init
    scene = state.simulator_state
    large = next(i for i in state.get_objects(env.item_type)
                 if scene.item_sizes[i.name] == "large")
    robot = state.get_objects(env.robot_type)[0]
    region = next(r for r in state.get_objects(env.region_type)
                  if r.name == "box0_left")
    nsrt = next(n for n in get_gt_nsrts("packing", env.predicates,
                                        get_gt_options("packing"))
                if n.option.name == "PlaceLarge")
    params = nsrt.sampler(state, set(), np.random.default_rng(0),
                          [robot, large, region])
    assert nsrt.option.params_space.contains(params)
