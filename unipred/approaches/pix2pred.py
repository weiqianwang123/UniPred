"""Pix2Pred (Table III): a vision-language model invents predicates and
labels their atoms in images; operators are learned from the labelled
demonstrations by effect extraction and intersection, without neural
predicate classifiers.

1. Predicates: the VLM proposes predicates (name, argument types, meaning)
   from the domain description, the types, the skills, the known predicates
   and images of a demonstration. With --pix2pred_gt_predicates, the
   names and types of the environment's hidden ground-truth predicates are
   used instead (the VLM still labels them).
2. Labels: the VLM lists the true atoms of the predicates in the image of a
   state; a predicate's classifier looks its atom up there (each image is
   labelled once). With --pix2pred_gt_label_predicates (comma-separated
   names), the atoms of those predicates come from the annotations of the
   recordings instead, in learning and in planning.
3. Operators: learned from the labelled demonstrations with
   --strips_learner (cluster_and_intersect in the configs); samplers are
   not needed offline.

Planning labels the image of a test scene the same way and plans a skeleton
without a simulator; it is scored by offline planning
(unipred/eval/offline_planning.py).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import dill as pkl
from gymnasium.spaces import Box

from predicators import utils
from predicators.approaches.bilevel_learning_approach import \
    read_predicate_config
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.envs import get_or_create_env
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.settings import CFG
from predicators.structs import NSRT, Dataset, DummyPredicate, Object, \
    ParameterizedOption, Predicate, State, Task, Type
from unipred.approaches.vila import objects_text, scene_image
from unipred.envs.offline_real import RecordedScene
from unipred.llm.client import LLMClient
from unipred.llm.pddl import load_prompt

AtomTuple = Tuple[str, Tuple[str, ...]]
# Images of the first demonstration shown when proposing predicates.
_NUM_PROPOSAL_IMAGES = 4


class Pix2PredApproach(BilevelPlanningApproach):
    """Predicates labelled by a VLM, operators by intersection (see module
    doc)."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._description: str = read_predicate_config(
            CFG.neupi_pred_config).get("description", "")
        self._gt_label_names = {
            n.strip()
            for n in CFG.pix2pred_gt_label_predicates.split(",") if n.strip()
        }
        self._client = LLMClient.from_settings(
            log_path=f"{utils.get_approach_save_path_str()}.llm_log.jsonl")
        # Name -> (types, meaning) of the predicates the VLM labels.
        self._specs: Dict[str, Tuple[List[str], str]] = {}
        self._predicates: Set[Predicate] = set()
        self._nsrts: Set[NSRT] = set()
        # Image -> the labelled true atoms.
        self._labels: Dict[str, FrozenSet[AtomTuple]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "pix2pred"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._predicates

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        demos = [t for t in dataset.trajectories if t.is_demo]
        if CFG.pix2pred_gt_predicates:
            env = get_or_create_env(CFG.env)
            hidden = env.predicates - self._initial_predicates
            self._specs = {
                p.name: ([t.name for t in p.types], "")
                for p in sorted(hidden)
            }
        else:
            self._specs = self._propose_predicates(
                [scene_image(s) for s in demos[0].states[:_NUM_PROPOSAL_IMAGES]])
        unknown = self._gt_label_names - set(self._specs)
        if unknown:
            raise ValueError(f"pix2pred_gt_label_predicates names "
                             f"{sorted(unknown)}, which are not among the "
                             f"predicates {sorted(self._specs)}.")
        self._build_predicates()
        names = ", ".join(f"{n}({', '.join(t)})"
                          for n, (t, _) in sorted(self._specs.items()))
        logging.info(f"Pix2Pred predicates: {names}")
        self._nsrts, _, _ = learn_nsrts_from_data(
            dataset.trajectories,
            self._train_tasks,
            self._get_current_predicates(),
            self._initial_options,
            self._action_space,
            ground_atom_dataset=None,
            sampler_learner=CFG.sampler_learner,
            annotations=None)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}.pix2pred.json", "w", encoding="utf-8") as f:
            json.dump(self._specs, f, indent=1)
        # The learned predicates' classifiers ask this approach's VLM, so
        # the NSRTs are saved with placeholder predicates.
        with open(f"{save_path}.NSRTs", "wb") as f:
            pkl.dump(
                utils.replace_nsrt_predicates(
                    self._nsrts, {
                        p: DummyPredicate(p.name, p.types)
                        for p in self._predicates
                    }), f)

    def load(self) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}.pix2pred.json", "r", encoding="utf-8") as f:
            self._specs = {n: (t, m) for n, (t, m) in json.load(f).items()}
        self._build_predicates()
        with open(f"{save_path}.NSRTs", "rb") as f:
            dummy_nsrts = pkl.load(f)
        by_name = {p.name: p for p in self._predicates}
        self._nsrts = set(
            utils.replace_nsrt_predicates(
                dummy_nsrts, {
                    DummyPredicate(n, p.types): p
                    for n, p in by_name.items()
                }))

    def _build_predicates(self) -> None:
        types = {t.name: t for t in self._types}
        self._predicates = {
            Predicate(name, [types[t] for t in arg_types],
                      self._labelled(name))
            for name, (arg_types, _) in self._specs.items()
        }

    def _labelled(self, name: str):
        """The classifier of a predicate: whether its atom is labelled true
        in the state's image."""

        def _classifier(state: State, objects: Sequence[Object]) -> bool:
            return (name, tuple(o.name for o in objects)) in self._label(state)

        return _classifier

    def _label(self, state: State) -> FrozenSet[AtomTuple]:
        """The true atoms in a state's image: labelled by the VLM, or taken
        from the annotations for the GT-label predicates."""
        image = scene_image(state)
        if image in self._labels:
            return self._labels[image]
        vlm_names = sorted(set(self._specs) - self._gt_label_names)
        atoms: Set[AtomTuple] = set()
        if vlm_names:
            atoms |= self._vlm_labels(image, list(state), vlm_names)
        if self._gt_label_names:
            scene = state.simulator_state
            assert isinstance(scene, RecordedScene)
            if scene.atoms is None:
                raise ValueError(f"{image} has no annotated atoms for "
                                 "pix2pred_gt_label_predicates.")
            atoms |= {a for a in scene.atoms if a[0] in self._gt_label_names}
        self._labels[image] = frozenset(atoms)
        return self._labels[image]

    def _vlm_labels(self, image: str, objects: List[Object],
                    names: List[str]) -> Set[AtomTuple]:
        predicates = "\n".join(
            f"- {n}({', '.join(self._specs[n][0])})"
            + (f": {self._specs[n][1]}" if self._specs[n][1] else "")
            for n in names)
        user = load_prompt("pix2pred_labels_user.txt").substitute(
            description=self._description or "(none)",
            predicates=predicates,
            objects=objects_text(objects))
        reply = self._client.complete(
            load_prompt("pix2pred_labels_system.txt").substitute(),
            user,
            "pix2pred_labels",
            images=[image])
        types = {o.name: o.type.name for o in objects}
        atoms = set()
        for name, args in re.findall(r"([A-Za-z_]\w*)\s*\(([^()]*)\)",
                                     reply):
            arg_names = tuple(a.strip() for a in args.split(",") if a.strip())
            if name in names and [types.get(a) for a in arg_names
                                  ] == self._specs[name][0]:
                atoms.add((name, arg_names))
        return atoms

    def _propose_predicates(
            self, images: List[str]) -> Dict[str, Tuple[List[str], str]]:
        type_names = sorted(t.name for t in self._types)
        user = load_prompt("pix2pred_predicates_user.txt").substitute(
            description=self._description or "(none)",
            types=", ".join(type_names),
            skills="\n".join(
                f"- {o.name}({', '.join(t.name for t in o.types)})"
                for o in sorted(self._initial_options)),
            known="\n".join(
                f"- {p.name}({', '.join(t.name for t in p.types)})"
                for p in sorted(self._initial_predicates)) or "(none)")
        reply = self._client.complete(
            load_prompt("pix2pred_predicates_system.txt").substitute(),
            user,
            "pix2pred_predicates",
            images=images)
        known = {p.name for p in self._initial_predicates}
        specs: Dict[str, Tuple[List[str], str]] = {}
        for name, args, meaning in re.findall(
                r"^\W*([A-Za-z_]\w*)\s*\(([^()]*)\)\s*:?\s*(.*)$", reply,
                re.MULTILINE):
            arg_types = [a.strip() for a in args.split(",") if a.strip()]
            if name not in known and arg_types and all(
                    t in type_names for t in arg_types):
                specs.setdefault(name, (arg_types, meaning.strip()))
        if not specs:
            raise ValueError("The VLM proposed no usable predicates.")
        return specs
