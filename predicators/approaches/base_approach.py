"""Base class for an approach."""

import abc
from collections import defaultdict
from typing import Any, Callable, List, Set

import numpy as np
from gymnasium.spaces import Box

from predicators.settings import CFG
from predicators.structs import Action, Dataset, Metrics, \
    ParameterizedOption, Predicate, State, Task, Type
from predicators.utils import ExceptionWithInfo


class BaseApproach(abc.ABC):
    """Base approach."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        """All approaches are initialized with only the necessary information
        about the environment."""
        self._initial_predicates = initial_predicates
        self._initial_options = initial_options
        self._types = types
        self._action_space = action_space
        self._train_tasks = train_tasks
        self._metrics: Metrics = defaultdict(float)
        self._set_seed(CFG.seed)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this approach, used as the argument to
        `--approach`."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def is_learning_based(self) -> bool:
        """Does the approach learn from the training tasks?"""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Return a policy for the given task, within the given number of
        seconds.

        A policy maps states to low-level actions.
        """
        raise NotImplementedError("Override me!")

    def get_execution_monitoring_info(self) -> List[Any]:
        """Provide any info the execution monitoring system might need to
        perform its job.

        For instance, a planning based approach might provide a symbolic
        plan that the monitor can check is being followed precisely
        during execution.
        """
        return []

    def solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Light wrapper around the abstract self._solve().

        Checks that actions are in the action space.
        """
        pi = self._solve(task, timeout)

        def _policy(state: State) -> Action:
            assert isinstance(state, State)
            act = pi(state)
            assert self._action_space.contains(act.arr)
            return act

        return _policy

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rng."""
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._action_space.seed(seed)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """For learning-based approaches, learn whatever is needed from the
        given dataset.

        Also, save the results of learning so they can be loaded in the
        future via load().
        """

    def load(self) -> None:
        """Load what learn_from_offline_dataset() saved under
        utils.get_approach_load_path_str().

        Only called if self.is_learning_based.
        """

    @property
    def metrics(self) -> Metrics:
        """Return a dictionary of metrics, which can hold arbitrary evaluation
        information about the performance of this approach."""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset the metrics dictionary."""
        self._metrics = defaultdict(float)


class ApproachTimeout(ExceptionWithInfo):
    """Exception raised when approach.solve() times out."""


class ApproachFailure(ExceptionWithInfo):
    """Exception raised when approach.solve() fails to compute a policy."""
