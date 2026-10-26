"""Tests for the MuJoCo stand-in of the real robot (unipred/robot_sim);
skipped without MuJoCo or the simulator's assets. The picks grasp straight
from above, so no GraspGenX server is needed."""

import pytest

pytest.importorskip("mujoco")

# pylint: disable=wrong-import-position
from unipred.robot_sim.assets import MENAGERIE, SCANNED_OBJECTS, \
    assets_dir, require

try:
    require(MENAGERIE, "franka_fr3/fr3.xml")
    require(SCANNED_OBJECTS, "models")
except FileNotFoundError:
    pytest.skip(f"no simulator assets in {assets_dir()}",
                allow_module_level=True)

from unipred.robot_sim.table_clean_mujoco import MujocoRobot, SceneLayout, \
    TableCleanMujoco


@pytest.fixture(name="sim")
def _sim():
    sim = TableCleanMujoco(SceneLayout((("toy0", "table"), ), "box", 0,
                                       False),
                           seed=0,
                           image_size=(120, 160),
                           grasps="top_down")
    yield sim
    sim.close()


def test_scene_starts_as_its_layout(sim):
    """The atoms are the layout's, the objects agree with them, and every
    object has a box in the image."""
    assert sim.atoms() == sorted([("BoxNear", ("box", )),
                                  ("HandEmpty", ("robot", )),
                                  ("ToyOnTable", ("toy0", "table")),
                                  ("WiperInBox", ("wiper", "box"))])
    assert not sim.problems()
    assert sim.render().shape == (120, 160, 3)
    boxes = sim.boxes()
    assert set(boxes) == set(sim.objects())
    for x0, y0, x1, y1 in boxes.values():
        assert 0 <= x0 < x1 <= 160 and 0 <= y0 < y1 <= 120


def test_skills_follow_table_clean_rules(sim):
    """A skill whose conditions do not hold does nothing; pick and place
    move the toy into the tray."""
    assert not sim.execute("PlaceToyInBox", ["robot", "toy0", "box"])
    assert not sim.execute("PullBoxIn", ["robot", "box"])
    assert sim.execute("PickToyFromTable", ["robot", "toy0", "table"])
    assert ("HoldingToy", ("robot", "toy0")) in sim.atoms()
    assert not sim.execute("PickWiperFromBox", ["robot", "wiper", "box"])
    assert sim.execute("PlaceToyInBox", ["robot", "toy0", "box"])
    assert ("ToyInBox", ("toy0", "box")) in sim.atoms()
    assert not sim.problems()


def test_description_rebuilds_the_scene(sim):
    """from_description gives the same scene (toys and their places)."""
    again = TableCleanMujoco.from_description(sim.description(),
                                              grasps="top_down")
    try:
        assert again.toy_models == sim.toy_models
        assert again.atoms() == sim.atoms()
        assert abs(again.boxes()["toy0"][0] - sim.boxes()["toy0"][0] *
                   again.render().shape[1] / 160) < 2.0
    finally:
        again.close()


def test_robot_interface_records_frames(sim):
    """MujocoRobot captures the camera and runs skills."""
    robot = MujocoRobot(sim)
    robot.capture()
    robot.execute("PushBoxOut", ["robot", "box"])
    assert ("BoxFar", ("box", )) in sim.atoms()
    assert len(robot.frames) == 1
    assert not sim.problems()
