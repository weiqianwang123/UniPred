"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.datasets.demo_only import human_demonstrator_policy
from predicators.envs import BaseEnv, create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG, get_allowed_query_type_names
from predicators.structs import Action, DemonstrationQuery, \
    DemonstrationResponse, GroundAtomsHoldQuery, GroundAtomsHoldResponse, \
    InteractionRequest, LowLevelTrajectory, Observation, PathToStateQuery, \
    PathToStateResponse, Query, Response, State, Task


class Teacher:
    """The teacher can respond to queries of various types."""

    def __init__(self, train_tasks: Sequence[Task]) -> None:
        self._env = create_new_env(CFG.env)
        self._train_tasks = train_tasks
        env_options = get_gt_options(self._env.get_name())
        self._pred_name_to_pred = {
            pred.name: pred
            for pred in self._env.predicates
        }
        self._allowed_query_type_names = get_allowed_query_type_names()
        self._oracle_approach = OracleApproach(
            self._env.predicates,
            env_options,
            self._env.types,
            self._env.action_space, [],
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized)
        self._rng = np.random.default_rng(CFG.seed)

    def answer_query(self, state: State, query: Query) -> Response:
        """The key method that the teacher defines."""
        assert query.__class__.__name__ in self._allowed_query_type_names, \
            f"Disallowed query: {query}"
        if isinstance(query, GroundAtomsHoldQuery):
            return self._answer_GroundAtomsHold_query(state, query)
        if isinstance(query, DemonstrationQuery):
            return self._answer_Demonstration_query(state, query)
        if isinstance(query, PathToStateQuery):
            return self._answer_PathToState_query(state, query)
        raise NotImplementedError(f"Unrecognized query: {query}.")

    def _answer_GroundAtomsHold_query(
            self, state: State,
            query: GroundAtomsHoldQuery) -> GroundAtomsHoldResponse:
        holds = {}
        for ground_atom in query.ground_atoms:
            pred = self._pred_name_to_pred[ground_atom.predicate.name]
            holds[ground_atom] = pred.holds(state, ground_atom.objects)
        return GroundAtomsHoldResponse(query, holds)

    def _answer_Demonstration_query(
            self, state: State,
            query: DemonstrationQuery) -> DemonstrationResponse:
        # The query is asking for a demonstration from the current state to
        # the goal from the train task.
        goal = self._train_tasks[query.train_task_idx].goal
        task = Task(state, goal)
        termination_function = task.goal_holds

        if CFG.demonstrator == "oracle":
            try:
                policy = self._oracle_approach.solve(task, CFG.timeout)
            except (ApproachTimeout, ApproachFailure):
                return DemonstrationResponse(query, teacher_traj=None)

        else:  # pragma: no cover
            assert CFG.demonstrator == "human"
            # Disable all built-in keyboard shortcuts.
            keymaps = {k for k in plt.rcParams if k.startswith("keymap.")}
            for k in keymaps:
                plt.rcParams[k].clear()
            # Create the environment-specific method for turning events into
            # actions. This should also log instructions.
            event_to_action = self._env.get_event_to_action_fn()
            caption = f"Please demonstrate achieving the goal:\n{goal}"
            policy = functools.partial(human_demonstrator_policy, self._env,
                                       caption, event_to_action)

        traj, _ = utils.run_policy(
            policy,
            self._env,
            "train",
            query.train_task_idx,
            termination_function=termination_function,
            max_num_steps=CFG.horizon,
            do_env_reset=False,  # important!
            exceptions_to_break_on={
                utils.OptionExecutionFailure,
                utils.HumanDemonstrationFailure,
            })

        teacher_traj = LowLevelTrajectory(traj.states,
                                          traj.actions,
                                          _is_demo=True,
                                          _train_task_idx=query.train_task_idx)
        return DemonstrationResponse(query, teacher_traj)

    def _answer_PathToState_query(
            self, state: State,
            query: PathToStateQuery) -> PathToStateResponse:
        # Answering this query requires an environment-specific way to find
        # the option that leads from the state to the goal state. None of the
        # environments in this package implements one.
        del state, query  # unused
        raise NotImplementedError("PathToStateQuery is not supported for "
                                  f"env {CFG.env}.")


@dataclass
class TeacherInteractionMonitor(utils.LoggingMonitor):
    """Wraps the interaction between agent and teacher to include generating
    and answering queries."""
    _request: InteractionRequest
    _teacher: Teacher
    _responses: List[Optional[Response]] = field(init=False,
                                                 default_factory=list)
    _query_cost: float = field(init=False, default=0.0)

    @property
    def _teacher_env(self) -> BaseEnv:
        """Exposes the teacher's environment to this monitor."""
        # This is a temporary hack. When issue #1443 is resolved, this can be
        # removed. That issue will require a fairly significant amount of work.
        return self._teacher._env  # pylint: disable=protected-access

    def reset(self, train_or_test: str, task_idx: int) -> None:
        self._teacher_env.reset(train_or_test, task_idx)

    def observe(self, obs: Observation, action: Optional[Action]) -> None:
        assert isinstance(obs, State)
        assert obs.allclose(self._teacher_env.get_observation())
        if action is not None:
            self._teacher_env.step(action)
        state = obs
        query = self._request.query_policy(state)
        if query is None:
            self._responses.append(None)
        else:
            self._responses.append(self._teacher.answer_query(state, query))
            self._query_cost += query.cost

    def get_responses(self) -> List[Optional[Response]]:
        """Return the responses."""
        return self._responses

    def get_query_cost(self) -> float:
        """Return the query cost."""
        return self._query_cost


@dataclass
class TeacherInteractionMonitorWithVideo(TeacherInteractionMonitor,
                                         utils.VideoMonitor):
    """A monitor that wraps a TeacherInteractionMonitor to optionally also
    render every state and action encountered, if CFG.make_interaction_videos
    is True.

    The render_fn is generally env.render.
    """

    def observe(self, obs: Observation, action: Optional[Action]) -> None:
        if isinstance(obs, State):
            assert obs.allclose(self._teacher_env.get_observation())
        if action is not None:
            self._teacher_env.step(action)
        state = obs
        query = self._request.query_policy(state)
        if query is None:
            response = None
            caption = "None"
        else:
            response = self._teacher.answer_query(state, query)
            self._query_cost += query.cost
            caption = f"{response}, cost={query.cost}"
        self._responses.append(response)
        if CFG.make_interaction_videos:
            self._video.extend(self._render_fn(action, caption))
