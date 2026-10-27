"""Tests for the annotated video of the closed loop (unipred/loop_video.py)."""

import imageio
import numpy as np

from predicators.structs import Predicate, State, Type
from unipred.envs.offline_real import RecordedScene
from unipred.loop_video import LoopVideo, predicate_labels
from unipred.perception.scene import SceneObservation

_ROBOT = Type("robot", ["f0"])
_TOY = Type("toy", ["f0"])
_HELD = Predicate("HoldingToy", [_ROBOT, _TOY],
                  lambda s, o: (("HoldingToy", ("robot", o[1].name))
                                in s.simulator_state.atoms))


def _state(held: str) -> State:
    robot, toy0, toy1 = _ROBOT("robot"), _TOY("toy0"), _TOY("toy1")
    atoms = frozenset({("HoldingToy", ("robot", held))}) if held else \
        frozenset()
    return State({
        robot: np.zeros(1, dtype=np.float32),
        toy0: np.array([1.0 if held == "toy0" else 0.0], dtype=np.float32),
        toy1: np.array([1.0 if held == "toy1" else 0.0], dtype=np.float32),
    },
                 simulator_state=RecordedScene("", atoms))


def test_invented_predicates_are_named_after_ground_truth():
    """An invented predicate takes the name of the ground-truth predicate it
    agrees with most, with the agreement."""
    invented = Predicate("neural_b_robot_toy_0", [_ROBOT, _TOY],
                         lambda s, o: s.get(o[1], "f0") > 0.5)
    states = [_state(""), _state("toy0"), _state("toy1")]
    labels = predicate_labels([invented], [_HELD], states)
    assert labels == {invented: ("HoldingToy", 1.0)}


def test_a_ground_truth_predicate_names_one_invented_predicate():
    """Of two invented predicates that match the same ground-truth
    predicate (one its negation), only the better match is named."""
    held = Predicate("neural_b_robot_toy_0", [_ROBOT, _TOY],
                     lambda s, o: s.get(o[1], "f0") > 0.5)
    free = Predicate(
        "neural_b_robot_toy_1", [_ROBOT, _TOY],
        lambda s, o: o[1].name == "toy1" and s.get(o[1], "f0") < 0.5)
    states = [_state(""), _state("toy0"), _state("toy1")]
    labels = predicate_labels([held, free], [_HELD], states)
    assert labels == {held: ("HoldingToy", 1.0)}


def test_video_of_keyframes_and_skills(tmp_path):
    """A keyframe is held for a while, a skill's frames follow; the frames
    show the image beside the panel."""
    invented = Predicate("neural_b_robot_toy_0", [_ROBOT, _TOY],
                         lambda s, o: s.get(o[1], "f0") > 0.5)
    video = LoopVideo("test", {invented}, {invented: ("HoldingToy", 0.95)},
                      fps=4,
                      hold=1.0)
    image = np.full((60, 80, 3), 200, dtype=np.uint8)
    observation = SceneObservation({"toy0": (10, 10, 20, 20)},
                                   {"toy0": None}, {}, {})
    video.keyframe(image, observation, {"toy0": "toy"}, _state("toy0"),
                   ["PickToyFromTable(robot, toy0, table)"], 0)
    video.skill([image, image], ["PickToyFromTable(robot, toy0, table)"], 0)
    assert len(video.frames) == 4 + 2
    assert video.frames[0].shape[1] > image.shape[1]
    path = tmp_path / "loop.mp4"
    video.save(str(path))
    assert imageio.get_reader(str(path)).count_frames() == 4 + 2
