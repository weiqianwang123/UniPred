"""Real-robot domains, built offline from processed recordings.

examples/real_robot/process_demos.py turns recorded sessions into
processed.pkl in a data directory: for every keyframe of every session, the
feature vector of each object (from its image crop) and, where annotated,
the ground-truth atoms. An OfflineRealEnv reads it:

- types: those of the domain spec, each with the crop features f0, f1, ...;
- predicates: those of the domain spec, read from the annotations of a
  keyframe (derived predicates are computed from them by the closure of the
  ground-truth operators). Annotations are required where the approaches
  and the evaluation need them: the given (non-excluded) predicates in
  demonstrations, and the initial atoms of test scenes. In a state without
  annotations (a new image, at deployment), a predicate is evaluated by a
  classifier trained on the annotated keyframes of the train sessions,
  over the features of its arguments;
- train tasks: the initial keyframes and goals of the train sessions, whose
  recorded demonstrations are the dataset (--offline_data_method recorded);
- test tasks: the test scenes (with --in_domain_test, the initial keyframes
  of the train sessions);
- no simulator: an approach is evaluated by offline planning
  (unipred/eval/offline_planning.py), which checks the skeleton it plans for
  a test scene against the ground-truth operators.

The options and ground-truth operators are those of the domain's
ground-truth models, registered for the env's name. A State's
simulator_state is the RecordedScene of the keyframe: its image (for the
VLM baselines) and annotated atoms (for the ground-truth predicates).

A domain is a subclass that names the env and the setting of its data
directory (see unipred/envs/table_clean_real.py).
"""

from __future__ import annotations

import itertools
import logging
import os
import pickle
from dataclasses import dataclass
from typing import ClassVar, Dict, FrozenSet, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from PIL import Image

from predicators import utils
from predicators.envs import BaseEnv
from predicators.ground_truth_models import get_gt_atoms_closure, \
    get_gt_options
from predicators.ml_models import MLPBinaryClassifier
from predicators.settings import CFG
from predicators.structs import Action, Dataset, EnvironmentTask, \
    GroundAtom, LowLevelTrajectory, Metrics, Object, ParameterizedOption, \
    Predicate, State, Type, _Option

AtomTuple = Tuple[str, Tuple[str, ...]]


@dataclass(frozen=True)
class RecordedScene:
    """What a keyframe keeps besides the object features."""
    # The keyframe's image file.
    image: str
    # The annotated ground-truth atoms, or None if not annotated.
    atoms: Optional[FrozenSet[AtomTuple]]


class OfflineRealEnv(BaseEnv):
    """A real-robot domain built from processed recordings (see module
    doc)."""
    # The setting that holds the data directory, e.g.
    # "table_clean_real_data_dir".
    data_dir_setting: ClassVar[str] = ""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        data_dir = getattr(CFG, self.data_dir_setting)
        path = os.path.join(data_dir, "processed.pkl")
        if not data_dir or not os.path.exists(path):
            raise FileNotFoundError(
                f"{self.get_name()} needs processed recordings: set "
                f"--{self.data_dir_setting} to the output directory of "
                f"examples/real_robot/process_demos.py (no {path}).")
        with open(path, "rb") as f:
            processed = pickle.load(f)
        self._data_dir = data_dir
        domain = processed["domain"]
        if domain["env"] != self.get_name():
            raise ValueError(f"{path} was processed for {domain['env']}, "
                             f"not {self.get_name()}.")
        features = [f"f{i}" for i in range(processed["feature_dim"])]
        self._types = {t: Type(t, features) for t in domain["types"]}
        self._predicates = {
            name: Predicate(name, [self._types[t] for t in types],
                            self._annotated(name))
            for name, types in domain["predicates"].items()
        }
        self._goal_names = set(domain["goal_predicates"])
        self._derived_names = set(domain.get("derived_predicates", []))
        self._sessions: List[Dict] = processed["sessions"]
        self.encoder: str = processed["encoder"]
        self.box_features: bool = processed.get("box_features", False)
        self._closure = get_gt_atoms_closure(self.get_name(), self.predicates)
        self._learned: Dict[str, MLPBinaryClassifier] = {}

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError(
            f"{self.get_name()} is built from recordings and has no "
            "simulator; it is evaluated by offline planning.")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [
            self._task(s, 0)
            for s in self._split("train", CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            sessions = self._split("train", CFG.num_test_tasks)
        else:
            sessions = self._split("test", CFG.num_test_tasks)
        return [self._task(s, 0) for s in sessions]

    @property
    def predicates(self) -> Set[Predicate]:
        return set(self._predicates.values())

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._predicates[n] for n in self._goal_names}

    @property
    def derived_predicates(self) -> Set[Predicate]:
        return {self._predicates[n] for n in self._derived_names}

    @property
    def types(self) -> Set[Type]:
        return set(self._types.values())

    @property
    def action_space(self) -> Box:
        # Actions are never executed; demonstrations carry their options.
        return Box(0, 1, (1, ))

    def get_recorded_dataset(self) -> Dataset:
        options = {o.name: o for o in get_gt_options(self.get_name())}
        trajectories = []
        for idx, session in enumerate(self._split("train",
                                                  CFG.num_train_tasks)):
            objects = self._objects(session)
            states = [
                self._state(session, i)
                for i in range(len(session["keyframes"]))
            ]
            actions = []
            for option_name, args in session["actions"]:
                if option_name not in options:
                    raise ValueError(f"{session['name']}: unknown option "
                                     f"{option_name}; the options are "
                                     f"{sorted(options)}.")
                option = _recorded(options[option_name],
                                   [objects[a] for a in args])
                actions.append(
                    Action(np.zeros(1, dtype=np.float32), option))
            trajectories.append(
                LowLevelTrajectory(states,
                                   actions,
                                   _is_demo=True,
                                   _train_task_idx=idx))
        return Dataset(trajectories)

    def evaluate_offline(self, approach: object) -> Metrics:
        # pylint: disable=import-outside-toplevel
        from unipred.eval.offline_planning import evaluate
        return evaluate(self, approach,
                        [t.task for t in self.get_test_tasks()])

    def scene_state(self, objects: Dict[str, str],
                    features: Dict[str, np.ndarray], image: str) -> State:
        """The state of a new scene (no annotations): its objects (name ->
        type), their features and its image file."""
        return State(
            {
                Object(n, self._types[t]): np.asarray(features[n],
                                                      dtype=np.float32)
                for n, t in objects.items()
            },
            simulator_state=RecordedScene(image, None))

    def annotated_states(self) -> List[State]:
        """The annotated keyframes of the train sessions."""
        return [
            self._state(session, index)
            for session in self._split("train", CFG.num_train_tasks)
            for index, keyframe in enumerate(session["keyframes"])
            if keyframe["atoms"] is not None
        ]

    def keyframe_image(self, state: State) -> str:
        """The image file of a state's keyframe."""
        assert isinstance(state.simulator_state, RecordedScene)
        return state.simulator_state.image

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.imshow(np.asarray(Image.open(self.keyframe_image(state))))
        ax.axis("off")
        if caption is not None:
            ax.set_title(caption)
        return fig

    def _split(self, split: str, num: int) -> List[Dict]:
        sessions = [s for s in self._sessions if s["split"] == split]
        return sessions[:num]

    def _objects(self, session: Dict) -> Dict[str, Object]:
        return {
            name: Object(name, self._types[t])
            for name, t in session["objects"].items()
        }

    def _state(self, session: Dict, index: int) -> State:
        keyframe = session["keyframes"][index]
        objects = self._objects(session)
        atoms = keyframe["atoms"]
        if atoms is not None and self._closure is not None:
            ground = self._closure(
                {
                    GroundAtom(self._predicates[n], [objects[a] for a in args])
                    for n, args in atoms
                }, set(objects.values()))
            atoms = [(a.predicate.name, tuple(o.name for o in a.objects))
                     for a in ground]
        scene = RecordedScene(
            os.path.join(self._data_dir, keyframe["image"]),
            None if atoms is None else frozenset(atoms))
        return State(
            {
                objects[n]: np.asarray(v, dtype=np.float32)
                for n, v in keyframe["features"].items()
            },
            simulator_state=scene)

    def _task(self, session: Dict, index: int) -> EnvironmentTask:
        objects = self._objects(session)
        goal = {
            GroundAtom(self._predicates[name], [objects[a] for a in args])
            for name, args in session["goal"]
        }
        return EnvironmentTask(self._state(session, index), goal)

    def _annotated(self, name: str):
        """The classifier of a predicate: whether the atom is annotated in
        the keyframe, or, without annotations, what the learned classifier
        says."""

        def _classifier(state: State, objects: Sequence[Object]) -> bool:
            scene = state.simulator_state
            assert isinstance(scene, RecordedScene)
            if scene.atoms is not None:
                return (name, tuple(o.name for o in objects)) in scene.atoms
            if name in self._derived_names:
                basic = {
                    p
                    for n, p in self._predicates.items()
                    if n not in self._derived_names
                }
                atoms = utils.abstract(state, basic)
                if self._closure is not None:
                    atoms = self._closure(atoms, set(state))
                return GroundAtom(self._predicates[name], objects) in atoms
            features = np.concatenate([state[o] for o in objects])
            return self._learned_classifier(name).classify(features)

        return _classifier

    def _learned_classifier(self, name: str) -> MLPBinaryClassifier:
        """A classifier of a basic predicate over the features of its
        arguments, trained on the annotated keyframes of the train
        sessions."""
        if name in self._learned:
            return self._learned[name]
        pred = self._predicates[name]
        inputs, labels = [], []
        for session in self._split("train", CFG.num_train_tasks):
            objects = self._objects(session)
            by_type = {
                t: [o for o in objects.values() if o.type == t]
                for t in pred.types
            }
            for index, keyframe in enumerate(session["keyframes"]):
                if keyframe["atoms"] is None:
                    continue
                state = self._state(session, index)
                for args in itertools.product(*(by_type[t]
                                                for t in pred.types)):
                    if len(set(args)) < len(args):
                        continue
                    inputs.append(np.concatenate([state[o] for o in args]))
                    labels.append(
                        (name, tuple(o.name for o in args)) in keyframe["atoms"])
        if len(set(labels)) < 2:
            raise ValueError(f"Cannot learn {name} from the annotations: it "
                             "needs true and false annotated atoms.")
        classifier = MLPBinaryClassifier(seed=CFG.seed,
                                         balance_data=True,
                                         max_train_iters=1000,
                                         learning_rate=1e-3,
                                         n_iter_no_change=1000,
                                         hid_sizes=[64],
                                         n_reinitialize_tries=1,
                                         weight_init="default")
        classifier.fit(np.array(inputs), np.array(labels))
        logging.info(f"Trained a classifier of {name} on {len(labels)} "
                     "annotated groundings.")
        self._learned[name] = classifier
        return classifier


def _recorded(option: ParameterizedOption,
              objects: Sequence[Object]) -> _Option:
    """An option as recorded. It ran to its end within one recorded step,
    so it is initiable and terminal in any state; the option's own
    functions may need features that recordings lack. Recordings have no
    parameters: the middle of the parameter space stands for them."""
    space = option.params_space
    recorded = option.ground(objects,
                             ((space.low + space.high) / 2).astype(np.float32))
    recorded.initiable = lambda state: True
    recorded.terminal = lambda state: True
    return recorded
