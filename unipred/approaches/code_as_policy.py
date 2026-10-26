"""Code as Policy (Table VI, "LLM propose code"): the LLM writes a Python
policy that chooses the robot's skills from low-level states.

The LLM sees the domain description, the object types and their features,
the skills (options) with their argument types and parameter ranges, the
goal predicates and a few demonstrations: initial low-level states, goals
and the skills used, with their parameters and the features they changed.
It writes ``policy(state, goal)`` (see
unipred/llm/prompts/code_as_policy_user.txt). To solve a task, the policy is
called whenever the current skill has finished, and the skill it returns is
executed.

Before testing, the policy is run on the training tasks with the option
model. If it raises an error or returns an invalid skill, the code and the
error are shown to the LLM, which writes a new version, at most
code_as_policy_max_repairs times. Not reaching a goal is not an error.
Every version is saved as <approach save path>.policy_<k>.py and the one
used for testing as <approach save path>.policy.py.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import signal
import threading
import traceback
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, \
    Set

import numpy as np
from gymnasium.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, BaseApproach
from predicators.approaches.bilevel_learning_approach import \
    read_predicate_config
from predicators.option_model import create_option_model
from predicators.settings import CFG
from predicators.structs import Action, Dataset, DummyOption, GroundAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type, \
    _Option
from unipred.llm.client import LLMClient
from unipred.llm.pddl import load_prompt

PolicyFunction = Callable[..., Any]

# Features shown per object in the prompt (the policy gets all of them),
# and feature changes shown per demonstrated skill.
_MAX_SHOWN_FEATURES = 12
_MAX_SHOWN_CHANGES = 8


class PolicyError(Exception):
    """The generated code fails: it raises an error, does not define the
    policy, or the policy returns an invalid skill."""


def extract_code(reply: str) -> str:
    """The Python code in an LLM reply: its first fenced code block, or the
    whole reply."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    return (match.group(1) if match else reply).strip() + "\n"


def compile_policy(code: str) -> PolicyFunction:
    """Run the code and return the policy function it defines."""
    namespace: Dict[str, Any] = {"math": math, "np": np}
    try:
        exec(compile(code, "<policy>", "exec"), namespace)  # pylint: disable=exec-used
    except Exception as err:  # pylint: disable=broad-except
        raise PolicyError(f"the code does not run: {_describe(err)}") from err
    policy = namespace.get("policy")
    if not callable(policy):
        raise PolicyError("the code does not define policy(state, goal).")
    return policy


def _describe(err: BaseException) -> str:
    """The error and the line of the generated code where it happened."""
    lines = [
        frame.lineno for frame in traceback.extract_tb(err.__traceback__)
        if frame.filename == "<policy>"
    ]
    where = f" (line {lines[-1]})" if lines else ""
    return "".join(traceback.format_exception_only(type(err),
                                                   err)).strip() + where


@contextlib.contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Raise TimeoutError in the with-block after the given number of
    seconds. Signals only work in the main thread; elsewhere there is no
    limit."""
    if seconds <= 0 or threading.current_thread() is not \
            threading.main_thread():
        yield
        return

    def _raise(signum: int, frame: Any) -> None:
        del signum, frame  # unused
        raise TimeoutError(f"the policy took longer than {seconds} s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _number(value: float) -> str:
    return f"{value:.3g}"


class CodeAsPolicyApproach(BaseApproach):
    """An LLM-written policy over low-level states (see module doc)."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # The domain description comes from the predicate config.
        self._description: str = read_predicate_config(
            CFG.neupi_pred_config).get("description", "")
        self._options = {o.name: o for o in initial_options}
        self._policy_fn: Optional[PolicyFunction] = None

    @classmethod
    def get_name(cls) -> str:
        return "code_as_policy"

    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        save_path = utils.get_approach_save_path_str()
        client = LLMClient.from_settings(log_path=f"{save_path}.llm_log.jsonl")
        system = load_prompt("code_as_policy_system.txt").substitute()
        prompt = self._prompt_fields(dataset.trajectories)
        repair = ""
        code = ""
        for version in range(CFG.code_as_policy_max_repairs + 1):
            user = load_prompt("code_as_policy_user.txt").substitute(
                **prompt, repair=repair)
            code = extract_code(client.complete(system, user,
                                                "code_as_policy"))
            with open(f"{save_path}.policy_{version}.py",
                      "w",
                      encoding="utf-8") as f:
                f.write(code)
            try:
                policy_fn = compile_policy(code)
                num_solved = self._run_on_train_tasks(policy_fn)
            except PolicyError as err:
                logging.info(f"Policy version {version} failed on the "
                             f"training tasks: {err}")
                repair = (f"\n# Your previous code\n```python\n{code}```\n"
                          f"It failed on a training task: {err}\n"
                          "Fix the code.\n")
                continue
            logging.info(f"Policy version {version} solves {num_solved} of "
                         f"{len(self._train_tasks)} training tasks.")
            break
        with open(f"{save_path}.policy.py", "w", encoding="utf-8") as f:
            f.write(code)
        self._set_policy(code)
        logging.info(f"LLM usage: {client.usage}")

    def load(self) -> None:
        with open(f"{utils.get_approach_load_path_str()}.policy.py",
                  "r",
                  encoding="utf-8") as f:
            self._set_policy(f.read())

    def _set_policy(self, code: str) -> None:
        try:
            self._policy_fn = compile_policy(code)
        except PolicyError as err:
            logging.info(f"The final policy cannot be used: {err}")
            self._policy_fn = None

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        del timeout  # the policy is executed, not searched
        if self._policy_fn is None:
            raise ApproachFailure("The LLM wrote no runnable policy.")
        policy_fn = self._policy_fn
        cur_option = DummyOption

        def _policy(state: State) -> Action:
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                try:
                    cur_option = self._next_option(policy_fn, state,
                                                   task.goal)
                except PolicyError as err:
                    raise ApproachFailure(
                        f"The generated policy failed: {err}") from err
                if not cur_option.initiable(state):
                    raise ApproachFailure(
                        f"The generated policy chose {cur_option.name}"
                        f"{cur_option.objects}, which cannot start here.")
            return cur_option.policy(state)

        return _policy

    def _run_on_train_tasks(self, policy_fn: PolicyFunction) -> int:
        """Run the policy on the training tasks with the option model and
        return how many it solves; raise PolicyError at the first
        error."""
        option_model = create_option_model(CFG.option_model_name)
        num_solved = 0
        for task in self._train_tasks:
            state = task.init
            for _ in range(CFG.horizon):
                if task.goal_holds(state):
                    break
                option = self._next_option(policy_fn, state, task.goal)
                if not option.initiable(state):
                    break
                state, num_actions = \
                    option_model.get_next_state_and_num_actions(state, option)
                if num_actions == 0:
                    break
            num_solved += task.goal_holds(state)
        return num_solved

    def _next_option(self, policy_fn: PolicyFunction, state: State,
                     goal: Set[GroundAtom]) -> _Option:
        """Call the policy and ground the skill it returns."""
        objects = {o.name: o for o in state}
        state_dict = {
            o.name: {
                "type": o.type.name,
                **{f: float(state.get(o, f))
                   for f in o.type.feature_names}
            }
            for o in state
        }
        goal_list = [(a.predicate.name, [o.name for o in a.objects])
                     for a in sorted(goal)]
        try:
            with _time_limit(CFG.code_as_policy_call_timeout):
                result = policy_fn(state_dict, goal_list)
        except Exception as err:  # pylint: disable=broad-except
            raise PolicyError(f"the policy raised {_describe(err)}") from err
        try:
            name, object_names, params = result
        except (TypeError, ValueError) as err:
            raise PolicyError(f"the policy returned {result!r} instead of "
                              "(skill_name, object_names, parameters).") \
                from err
        if name not in self._options:
            raise PolicyError(f"there is no skill {name!r}.")
        option = self._options[name]
        if not isinstance(object_names, (list, tuple)) or any(
                n not in objects for n in object_names):
            raise PolicyError(f"{name} got the objects {object_names!r}, "
                              "which are not all object names.")
        args = [objects[n] for n in object_names]
        if len(args) != len(option.types) or not all(
                o.is_instance(t) for o, t in zip(args, option.types)):
            raise PolicyError(
                f"{name} takes objects of the types "
                f"({', '.join(t.name for t in option.types)}), not "
                f"{list(object_names)}.")
        space = option.params_space
        try:
            values = np.asarray(params, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as err:
            raise PolicyError(f"{name} got the parameters {params!r}, which "
                              "are not numbers.") from err
        if values.shape != space.shape or not np.all(np.isfinite(values)):
            raise PolicyError(f"{name} takes {space.shape[0]} finite "
                              f"parameters, not {params!r}.")
        return option.ground(args, np.clip(values, space.low, space.high))

    def _prompt_fields(
            self, trajectories: Sequence[LowLevelTrajectory]) -> Dict[str, str]:
        """The parts of the prompt that do not change between versions."""
        types = sorted(self._types, key=lambda t: t.name)
        type_lines = [
            f"- {t.name}: {', '.join(self._shown_features(t)) or '(none)'}"
            for t in types
        ]
        skill_lines = []
        for option in sorted(self._options.values(), key=lambda o: o.name):
            space = option.params_space
            if space.shape[0] == 0:
                params = "no parameters"
            else:
                ranges = ", ".join(
                    f"[{_number(lo)}, {_number(hi)}]"
                    for lo, hi in zip(space.low, space.high))
                params = f"parameters in {ranges}"
            skill_lines.append(
                f"- {option.name}({', '.join(t.name for t in option.types)})"
                f": {params}")
        goal_predicates = sorted(
            {a.predicate
             for task in self._train_tasks for a in task.goal},
            key=lambda p: p.name)
        goal_lines = [
            f"- {p.name}({', '.join(t.name for t in p.types)})"
            for p in goal_predicates
        ]
        demos = [t for t in trajectories
                 if t.is_demo][:CFG.unipred_num_demo_prompts]
        example = min(self._train_tasks[0].init, key=lambda o: o.name)
        return {
            "description": self._description or "(none)",
            "types": "\n".join(type_lines),
            "skills": "\n".join(skill_lines),
            "goal_predicates": "\n".join(goal_lines) or "(none)",
            "demos": "\n".join(
                self._render_demo(k, traj) for k, traj in enumerate(demos))
            or "(none)",
            "example_object": example.name,
            "example_feature": (example.type.feature_names or ["type"])[0],
        }

    @staticmethod
    def _shown_features(object_type: Type) -> List[str]:
        names = list(object_type.feature_names)
        if len(names) <= _MAX_SHOWN_FEATURES:
            return names
        return names[:_MAX_SHOWN_FEATURES] + [
            f"... ({len(names) - _MAX_SHOWN_FEATURES} more, up to "
            f"{names[-1]})"
        ]

    def _render_demo(self, k: int, traj: LowLevelTrajectory) -> str:
        task = self._train_tasks[traj.train_task_idx]
        init = traj.states[0]
        lines = [f"Demonstration {k + 1}:", "  initial state:"]
        for obj in sorted(init, key=lambda o: o.name):
            values = [
                f"{f}={_number(init.get(obj, f))}"
                for f in obj.type.feature_names[:_MAX_SHOWN_FEATURES]
            ]
            lines.append(f"    {obj.name} ({obj.type.name}): "
                         f"{', '.join(values)}")
        goal = ", ".join(
            f"{a.predicate.name}({', '.join(o.name for o in a.objects)})"
            for a in sorted(task.goal))
        lines += [f"  goal: {goal}", "  skills:"]
        # The index of the first action of each option, and of the state
        # after it.
        starts = [
            i for i, action in enumerate(traj.actions)
            if i == 0 or action.get_option() is not traj.actions[i -
                                                                 1].get_option()
        ]
        ends = starts[1:] + [len(traj.actions)]
        for start, end in zip(starts, ends):
            option = traj.actions[start].get_option()
            params = ", ".join(_number(p) for p in option.params)
            changes = self._changes(traj.states[start], traj.states[end])
            lines.append(
                f"    {option.name}({', '.join(o.name for o in option.objects)})"
                f" [{params}] -> {changes}")
        return "\n".join(lines)

    @staticmethod
    def _changes(before: State, after: State) -> str:
        changes = []
        for obj in sorted(before, key=lambda o: o.name):
            for feature in obj.type.feature_names:
                old, new = before.get(obj, feature), after.get(obj, feature)
                if abs(new - old) > 1e-3:
                    changes.append(f"{obj.name}.{feature} {_number(old)} -> "
                                   f"{_number(new)}")
        if len(changes) > _MAX_SHOWN_CHANGES:
            changes = changes[:_MAX_SHOWN_CHANGES] + [
                f"... ({len(changes) - _MAX_SHOWN_CHANGES} more)"
            ]
        return "; ".join(changes) or "no change"
