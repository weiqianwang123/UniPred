"""Tests for the GNN NSRT policy approach."""

from predicators import utils
from predicators.approaches.gnn_nsrt_policy_approach import \
    GNNNSRTPolicyApproach
from predicators.datasets import create_dataset
from predicators.envs.satellites import SatellitesEnv
from predicators.ground_truth_models import get_gt_options


def test_input_features_cover_all_known_predicates():
    """The graph encoding has a feature for every known predicate and type,
    also those that no demonstration contains (a test task may)."""
    utils.reset_config({
        "env": "satellites",
        "num_train_tasks": 1,
        "gnn_num_epochs": 1,
        "sesame_task_planner": "astar",
    })
    env = SatellitesEnv()
    options = get_gt_options(env.get_name())
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = GNNNSRTPolicyApproach(env.predicates, options, env.types,
                                     env.action_space, train_tasks)
    approach.learn_from_offline_dataset(
        create_dataset(env, train_tasks, options))
    features = approach._node_feature_to_index  # pylint: disable=protected-access
    for pred in env.predicates:
        if pred.arity == 1:
            assert pred in features
    for obj_type in env.types:
        assert f"type_{obj_type.name}" in features
