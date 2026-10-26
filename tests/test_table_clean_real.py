"""Tests for Table Clean Real, built from the processed synthetic fixture,
and for offline planning evaluation."""

import dataclasses
import os

import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_atoms_closure, \
    get_gt_nsrts, get_gt_options
from unipred.envs.offline_real import RecordedScene
from unipred.eval.offline_planning import check_skeleton, evaluate


def _env(processed, **settings):
    utils.reset_config({
        "env": "table_clean_real",
        "table_clean_real_data_dir": processed,
        "num_train_tasks": 4,
        "num_test_tasks": 2,
        "offline_data_method": "recorded",
        "bilevel_plan_without_sim": True,
        "sesame_task_planner": "astar",
        **settings
    })
    return create_new_env("table_clean_real")


def test_tasks_and_recorded_dataset(processed):
    """Train tasks are the recorded sessions, whose demonstrations are the
    dataset; test tasks are the held-out scenes (or, in domain, the train
    sessions)."""
    env = _env(processed)
    assert len(env.get_train_tasks()) == 4
    assert len(env.get_test_tasks()) == 2
    dataset = env.get_recorded_dataset()
    assert len(dataset.trajectories) == 4
    traj = dataset.trajectories[0]
    assert len(traj.states) == len(traj.actions) + 1 == 10
    option = traj.actions[0].get_option()
    assert option.name == "PickToyFromTable"
    assert option.initiable(traj.states[0]) and option.terminal(
        traj.states[1])
    assert isinstance(traj.states[0].simulator_state, RecordedScene)
    with pytest.raises(NotImplementedError, match="no simulator"):
        env.simulate(traj.states[0], traj.actions[0])
    env = _env(processed, in_domain_test=True)
    assert [t.init.simulator_state for t in env.get_test_tasks()
            ] == [t.init.simulator_state for t in env.get_train_tasks()[:2]]


def test_annotated_and_learned_predicates(processed):
    """Predicates are read from annotations (derived ones computed by the
    closure); without annotations, a learned classifier decides."""
    env = _env(processed)
    task = env.get_test_tasks()[0].task
    atoms = utils.abstract(task.init, env.predicates)
    names = {a.predicate.name for a in atoms}
    assert "HandEmpty" in names or "HoldingToy" in names \
        or "HoldingWiper" in names
    unannotated = task.init.copy()
    unannotated.simulator_state = dataclasses.replace(
        task.init.simulator_state, atoms=None)
    toy_on_table = {p for p in env.predicates if p.name == "ToyOnTable"}
    learned = utils.abstract(unannotated, toy_on_table)
    assert all(a.predicate.name == "ToyOnTable" for a in learned)


def test_oracle_plans_are_valid_and_errors_are_explained(processed):
    """The ground-truth operators solve every test scene offline; a wrong
    skeleton is rejected with the step that fails."""
    env = _env(processed)
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    tasks = [t.task for t in env.get_test_tasks()]
    metrics = evaluate(env, approach, tasks)
    assert metrics["num_solved"] == metrics["num_total"] == 2
    task = env.get_train_tasks()[0].task
    objects = {o.name: o for o in task.init}
    place = next(o for o in options if o.name == "PlaceToyInBox")
    problem = check_skeleton(
        [(place, [objects["robot"], objects["toy0"], objects["box"]])],
        utils.abstract(task.init, env.predicates), task.goal,
        list(task.init), get_gt_nsrts(env.get_name(), env.predicates,
                                      options),
        get_gt_atoms_closure(env.get_name(), env.predicates))
    assert problem is not None and "step 1" in problem \
        and "HoldingToy" in problem


def test_processed_crops_follow_the_annotated_objects(processed):
    """The processed boxes belong to the objects the annotations name: in
    every keyframe where the robot holds a toy, that toy's box is the toy
    box nearest the robot's (a fixture or recording that names its toys
    otherwise than left to right in the first keyframe swaps them)."""
    # pylint: disable=import-outside-toplevel
    import pickle
    with open(os.path.join(processed, "processed.pkl"), "rb") as f:
        sessions = pickle.load(f)["sessions"]
    checked = 0
    for session in sessions:
        toys = [n for n, t in session["objects"].items() if t == "toy"]
        for keyframe in session["keyframes"]:
            held = [
                args[1] for name, args in keyframe["atoms"] or []
                if name == "HoldingToy"
            ]
            if not held or len(toys) < 2:
                continue
            robot = keyframe["boxes"]["robot"]

            def _distance(name: str, robot=robot, keyframe=keyframe) -> float:
                box = keyframe["boxes"][name]
                return abs(box[0] + box[2] - robot[0] - robot[2]) + abs(
                    box[1] + box[3] - robot[1] - robot[3])

            assert min(toys, key=_distance) == held[0], (session["name"],
                                                         keyframe["image"])
            checked += 1
    assert checked > 0
