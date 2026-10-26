"""ViLa (Table III): a vision-language model plans the sequence of skills
directly from the image of a scene, without learned symbols.

Modes (--vila_mode):

- zero_shot: the VLM gets the domain description, the skills, the scene's
  objects and goal, and its image;
- few_shot: also a few demonstrations: the first image, objects, goal and
  skill sequence of recorded sessions (their images come first);
- hpe: few_shot plus hand-engineered constraints, written in the text file
  --vila_constraints (e.g. "the hand must be empty before picking").

The plan is evaluated offline (unipred/eval/offline_planning.py), through
get_last_skeleton(); it is never executed. The image of a scene is that of
its recorded keyframe (unipred/envs/offline_real.py).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Dict, List, Sequence, Set

from gymnasium.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, BaseApproach
from predicators.approaches.bilevel_learning_approach import \
    read_predicate_config
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type
from unipred.envs.offline_real import RecordedScene
from unipred.eval.offline_planning import Skeleton
from unipred.llm.client import LLMClient
from unipred.llm.pddl import load_prompt

_MODES = ("zero_shot", "few_shot", "hpe")


def scene_image(state: State) -> str:
    """The image file of a recorded scene."""
    if not isinstance(state.simulator_state, RecordedScene):
        raise ApproachFailure("The VLM baselines need recorded scenes with "
                              "images (an offline real-robot domain).")
    return state.simulator_state.image


def atoms_text(atoms: Set[GroundAtom]) -> str:
    """Atoms as Name(object, ...), sorted."""
    return ", ".join(
        sorted(f"{a.predicate.name}({', '.join(o.name for o in a.objects)})"
               for a in atoms))


def objects_text(objects: Sequence[Object]) -> str:
    """Objects as name (type), sorted."""
    return ", ".join(f"{o.name} ({o.type.name})"
                     for o in sorted(objects, key=lambda o: o.name))


def parse_skeleton(reply: str, options: Dict[str, ParameterizedOption],
                   objects: Dict[str, Object]) -> Skeleton:
    """The skills of a reply with one Name(object, ...) per line; raises
    ApproachFailure for an unknown skill or object or wrong types."""
    skeleton = []
    for name, args in re.findall(r"([A-Za-z_]\w*)\s*\(([^()]*)\)", reply):
        if name not in options:
            raise ApproachFailure(f"The VLM used the unknown skill {name}.")
        option = options[name]
        arg_names = [a.strip() for a in args.split(",") if a.strip()]
        if any(a not in objects for a in arg_names):
            raise ApproachFailure(
                f"The VLM used unknown objects in {name}({args}).")
        arg_objects = [objects[a] for a in arg_names]
        if len(arg_objects) != len(option.types) or not all(
                o.is_instance(t) for o, t in zip(arg_objects, option.types)):
            raise ApproachFailure(f"The VLM gave {name} objects of the wrong "
                                  f"types: {name}({args}).")
        skeleton.append((option, arg_objects))
    return skeleton


class VilaApproach(BaseApproach):
    """A VLM that plans skills from images (see module doc)."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        if CFG.vila_mode not in _MODES:
            raise ValueError(f"Unknown vila_mode {CFG.vila_mode!r}; use one "
                             f"of {', '.join(_MODES)}.")
        self._description: str = read_predicate_config(
            CFG.neupi_pred_config).get("description", "")
        self._options = {o.name: o for o in initial_options}
        self._constraints = ""
        if CFG.vila_mode == "hpe":
            if not CFG.vila_constraints:
                raise ValueError("vila_mode hpe needs --vila_constraints, a "
                                 "text file of hand-written constraints.")
            with open(CFG.vila_constraints, "r", encoding="utf-8") as f:
                self._constraints = f.read().strip()
        # Few-shot demonstrations: image, objects, goal and plan.
        self._demos: List[Dict[str, str]] = []
        self._client: LLMClient = LLMClient.from_settings(
            log_path=f"{utils.get_approach_save_path_str()}.llm_log.jsonl")
        self._last_skeleton: Skeleton = []

    @classmethod
    def get_name(cls) -> str:
        return "vila"

    @property
    def is_learning_based(self) -> bool:
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        if CFG.vila_mode != "zero_shot":
            for traj in [t for t in dataset.trajectories
                         if t.is_demo][:CFG.unipred_num_demo_prompts]:
                task = self._train_tasks[traj.train_task_idx]
                plan = []
                for i, action in enumerate(traj.actions):
                    option = action.get_option()
                    if i == 0 or option is not traj.actions[i -
                                                            1].get_option():
                        plan.append(f"{option.name}("
                                    f"{', '.join(o.name for o in option.objects)})")
                self._demos.append({
                    "image": scene_image(traj.states[0]),
                    "objects": objects_text(list(traj.states[0])),
                    "goal": atoms_text(task.goal),
                    "plan": "\n".join(plan),
                })
        with open(f"{utils.get_approach_save_path_str()}.vila_demos.json",
                  "w",
                  encoding="utf-8") as f:
            json.dump(self._demos, f, indent=1)

    def load(self) -> None:
        with open(f"{utils.get_approach_load_path_str()}.vila_demos.json",
                  "r",
                  encoding="utf-8") as f:
            self._demos = json.load(f)

    def get_last_skeleton(self) -> Skeleton:
        """The plan of the last call of solve()."""
        return self._last_skeleton

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        del timeout  # the VLM is asked once
        image = scene_image(task.init)
        skills = "\n".join(
            f"- {o.name}({', '.join(t.name for t in o.types)})"
            for o in sorted(self._options.values(), key=lambda o: o.name))
        constraints = (f"\nConstraints:\n{self._constraints}\n"
                       if self._constraints else "")
        demos = "\n".join(
            f"Demonstration {k + 1} (image {k + 1}):\n"
            f"Objects: {d['objects']}\nGoal: {d['goal']}\nPlan:\n{d['plan']}"
            for k, d in enumerate(self._demos)) or "(none)"
        user = load_prompt("vila_user.txt").substitute(
            description=self._description or "(none)",
            skills=skills,
            constraints=constraints,
            demos=demos,
            objects=objects_text(list(task.init)),
            goal=atoms_text(task.goal))
        reply = self._client.complete(
            load_prompt("vila_system.txt").substitute(),
            user,
            "vila",
            images=[d["image"] for d in self._demos] + [image])
        self._last_skeleton = parse_skeleton(reply, self._options,
                                             {o.name: o
                                              for o in task.init})
        logging.info(f"VLM plan: {reply.strip()}")

        def _policy(state: State) -> Action:
            raise ApproachFailure("ViLa's plans are only evaluated offline.")

        return _policy
