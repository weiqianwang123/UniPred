"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List, Set

from predicators.datasets.demo_only import create_demo_data
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Predicate, Task


def create_dataset(env: BaseEnv, train_tasks: List[Task],
                   known_options: Set[ParameterizedOption],
                   known_predicates: Set[Predicate]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks for
    an environment.

    Some or all of this data may be loaded from disk.
    """
    del known_predicates  # unused by the remaining dataset methods
    if CFG.offline_data_method == "demo":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=False)
    if CFG.offline_data_method == "demo+reward":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=False,
                                include_reward=True)
    if CFG.offline_data_method == "demo+gt_operators":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=True)
    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
