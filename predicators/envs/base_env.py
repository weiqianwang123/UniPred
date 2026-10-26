"""Base class for an environment."""

import abc
from typing import Any, List, Optional, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Action, Dataset, DefaultEnvironmentTask, \
    EnvironmentTask, GroundAtom, Metrics, Observation, Predicate, State, \
    Type, Video


class BaseEnv(abc.ABC):
    """Base environment."""

    def __init__(self, use_gui: bool = True) -> None:
        self._current_observation: Observation = None  # set in reset
        self._current_task = DefaultEnvironmentTask  # set in reset
        self._set_seed(CFG.seed)
        # These are generated lazily when get_train_tasks or get_test_tasks is
        # called. This is necessary because environment attributes are often
        # initialized in __init__ in subclasses, and super().__init__ needs
        # to be called in those subclasses first, to set the env seed.
        self._train_tasks: List[EnvironmentTask] = []
        self._test_tasks: List[EnvironmentTask] = []
        # If the environment has a GUI, this determines whether to launch it.
        self._using_gui = use_gui

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this environment, used as the argument to
        `--env`."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """Get the next state, given a state and an action.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.

        This function is primarily used in the default option model, and
        for implementing the default self.step(action). It is not meant to
        be part of the "final system", where the environment is the real world.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for training."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for testing / evaluation."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        raise NotImplementedError("Override me!")

    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        """Get the goal predicates that we want the agent to use, which may be
        different than the ones the demonstrator uses.

        This is used when inventing VLM predicates. Unless overridden,
        these are the same as the original goal predicates.
        """
        return self.goal_predicates

    @property
    def derived_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates whose truth values are derived
        from other predicates.

        No operator changes a derived predicate directly. When planning
        with them, install an atoms closure (see predicators/hooks.py)
        that recomputes them, and pass them to utils.create_pddl_domain()
        so that they are not written as operator effects. By default, an
        environment has no derived predicates.
        """
        return set()

    @property
    @abc.abstractmethod
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def action_space(self) -> Box:
        """Get the action space of this environment."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render a state and action into a Matplotlib figure.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        For environments which don't use Matplotlib for rendering, this
        function should be overriden to simply crash.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        raise NotImplementedError("Matplotlib rendering not implemented!")

    @property
    def using_gui(self) -> bool:
        """Whether the GUI for this environment is activated."""
        return self._using_gui

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        """Render a state and action into a list of images.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        By default, calls render_state_plt, but subclasses may override,
        e.g. if they do not use Matplotlib for rendering, and thus do not
        define a render_state_plt() function.
        """
        fig = self.render_state_plt(state, task, action, caption)
        img = utils.fig2data(fig, dpi=CFG.render_state_dpi)
        plt.close()
        return [img]

    def render_plt(self,
                   action: Optional[Action] = None,
                   caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render the current state and action into a Matplotlib figure.

        By default, calls render_state_plt, but subclasses may override.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        assert isinstance(self._current_observation, State), \
            "render_plt() only works in fully-observed environments."
        return self.render_state_plt(self._current_observation,
                                     self._current_task, action, caption)

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        """Render the current state and action into a list of images.

        By default, calls render_state, but subclasses may override.
        """
        assert isinstance(self._current_observation, State), \
            "render_state() only works in fully-observed environments."
        return self.render_state(self._current_observation, self._current_task,
                                 action, caption)

    def get_train_tasks(self) -> List[EnvironmentTask]:
        """Return the ordered list of tasks for training."""
        if not self._train_tasks:
            self._train_tasks = self._generate_train_tasks()
        return self._train_tasks

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """Return the ordered list of tasks for testing / evaluation."""
        if not self._test_tasks:
            self._test_tasks = self._generate_test_tasks()
        return self._test_tasks

    def get_recorded_dataset(self) -> Dataset:
        """The demonstrations of the train tasks, recorded outside a
        simulator (--offline_data_method recorded).

        Environments built from recorded data, such as real-robot
        domains, override this; their demonstrations cannot be
        generated by planning in a simulator.
        """
        raise NotImplementedError(
            f"{self.get_name()} has no recorded demonstrations.")

    def evaluate_offline(self, approach: Any) -> Optional[Metrics]:
        """Evaluate an approach (a BaseApproach) on the test tasks without
        executing its policies.

        Environments without a simulator override this and return the
        test metrics, with at least num_solved and num_total. By
        default, None: main.py executes the policies in the environment.
        """
        del approach  # unused
        return None

    @property
    def _current_state(self) -> State:
        """Default for environments where states are observations."""
        assert isinstance(self._current_observation, State)
        return self._current_observation

    def goal_reached(self) -> bool:
        """Default implementation assumes environment tasks are tasks.

        Subclasses may override.
        """
        # NOTE: this is a convenience hack because most environments that are
        # currently implemented have goal descriptions that are simply sets of
        # ground atoms. In the future, it may be better to implement this on a
        # per-environment basis anyway, to make clear that we do not need to
        # make this assumption about goal descriptions in general.
        goal = self._current_task.goal_description
        assert isinstance(goal, set)
        assert not goal or isinstance(next(iter(goal)), GroundAtom)
        return all(goal_atom.holds(self._current_state) for goal_atom in goal)

    def get_task(self, train_or_test: str, task_idx: int) -> EnvironmentTask:
        """Return the train or test task at the given index."""
        if train_or_test == "train":
            tasks = self.get_train_tasks()
        elif train_or_test == "test":
            tasks = self.get_test_tasks()
        else:
            raise ValueError(f"get_task called with invalid train_or_test: "
                             f"{train_or_test}.")
        return tasks[task_idx]

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rngs."""
        self._seed = seed
        # The train/test rng should be used when generating
        # train/test tasks respectively.
        self._train_rng = np.random.default_rng(self._seed)
        self._test_rng = np.random.default_rng(self._seed +
                                               CFG.test_env_seed_offset)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        # Copy to prevent external changes to the environment's state.
        # This default implementation of reset assumes that observations are
        # states. Subclasses with different states should override.
        assert isinstance(self._current_observation, State)
        return self._current_observation.copy()

    def step(self, action: Action) -> Observation:
        """Apply the action, update the state, and return an observation.

        Note that this action is a low-level action (i.e., action.arr
        is a member of self.action_space), NOT an option.

        By default, this function just calls self.simulate. However,
        environments that maintain a more complicated internal state,
        or that don't implement simulate(), may override this method.
        """
        assert isinstance(self._current_observation, State)
        self._current_observation = self.simulate(self._current_observation,
                                                  action)
        # Copy to prevent external changes to the environment's state.
        return self._current_observation.copy()

    def get_observation(self) -> Observation:
        """Get the current observation of this environment."""
        assert isinstance(self._current_observation, State)
        return self._current_observation.copy()
