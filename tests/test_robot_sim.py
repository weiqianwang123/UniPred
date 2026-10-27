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

from unipred.robot_sim.cluttered_retrieval_mujoco import ClutterLayout, \
    ClutteredRetrievalMujoco
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
    """MujocoRobot captures the camera, runs skills and records them."""
    robot = MujocoRobot(sim, video_period=0.5)
    assert robot.video() is None
    robot.capture()
    start = sim.data.time
    robot.execute("PushBoxOut", ["robot", "box"])
    assert ("BoxFar", ("box", )) in sim.atoms()
    assert len(robot.frames) == 1
    video = robot.video()
    assert len(video) >= int((sim.data.time - start) / 0.5)
    assert all(frame.shape == (120, 160, 3) for frame in video)
    assert sim.video is None
    assert not sim.problems()


def test_collected_sessions_name_toys_left_to_right(tmp_path):
    """collect_sim_demos.py writes a demonstration and a test scene in the
    raw format, the toys named left to right in the first keyframe."""
    # pylint: disable=import-outside-toplevel
    import json
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "examples/real_robot/collect_sim_demos.py", "--out",
        str(tmp_path), "--num_train", "1", "--num_test", "1", "--grasps",
        "top_down"
    ],
                   check=True)
    for name in ("train_00", "test_00"):
        session = tmp_path / name
        boxes = json.loads((session / "gt_boxes.json").read_text())
        toys = sorted(n for n in boxes[0] if n.startswith("toy"))
        centres = [boxes[0][t][0] + boxes[0][t][2] for t in toys]
        assert centres == sorted(centres)
        info = json.loads((session / "session.json").read_text())
        assert set(toys) <= set(info["models"])
        assert sorted(n for n, _ in info["sim"]["toys"]) == toys
    actions = json.loads((tmp_path / "train_00" / "actions.json").read_text())
    atoms = json.loads((tmp_path / "train_00" / "atoms.json").read_text())
    assert len(atoms) == len(actions) + 1
    assert (tmp_path / "train_00" / f"{len(actions)}.png").exists()
    # The video of every skill; a test scene has none.
    for k in range(len(actions)):
        assert list((tmp_path / "train_00" / "frames" / str(k)).glob("*.jpg"))
    assert not (tmp_path / "train_00" / "frames" /
                str(len(actions))).exists()
    assert not (tmp_path / "test_00" / "frames").exists()


def test_cluttered_retrieval_skills():
    """A blocked target cannot be picked; removing the bottle and pushing
    the bulky obstacle in front of it unblock it (and the camera no longer
    sees them hide it); then it goes into the basket."""
    layout = ClutterLayout(("target0", "target1"),
                           (("obstacle0", "crate", "target0", 0),
                            ("obstacle1", "bottle", "target0", 1),
                            ("obstacle2", "crate", "target1", 0)))
    with pytest.raises(ValueError, match="basket holds"):
        ClutteredRetrievalMujoco(ClutterLayout(
            tuple(f"target{k}" for k in range(4)), ()))
    sim = ClutteredRetrievalMujoco(layout, seed=0, grasps="top_down")
    try:
        atoms = set(sim.atoms())
        assert {("Blocks", ("obstacle0", "target0")),
                ("Blocks", ("obstacle1", "target0")),
                ("Bulky", ("obstacle0", )),
                ("Pickable", ("obstacle1", ))} <= atoms
        assert not sim.problems()
        assert not sim.execute("PickTarget", ["robot", "target0"])
        assert not sim.execute("RemoveObstacle",
                               ["robot", "obstacle0", "target0"])
        assert sim.execute("PushObstacle", ["robot", "obstacle0", "target0"])
        assert not sim.execute("PickTarget", ["robot", "target0"])
        assert sim.execute("RemoveObstacle",
                           ["robot", "obstacle1", "target0"])
        assert not any(a[0] == "Blocks" and a[1][1] == "target0"
                       for a in sim.atoms())
        assert not sim.problems()
        assert sim.execute("PickTarget", ["robot", "target0"])
        assert sim.execute("PlaceTargetInBox", ["robot", "target0", "box"])
        assert ("InBox", ("target0", "box")) in sim.atoms()
        assert not sim.problems()
    finally:
        sim.close()


def test_collected_cluttered_retrieval_session(tmp_path):
    """collect_sim_demos.py --domain cluttered_retrieval writes sessions
    whose targets and obstacles are named left to right and whose scene
    rebuilds from its description."""
    # pylint: disable=import-outside-toplevel
    import json
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "examples/real_robot/collect_sim_demos.py", "--out",
        str(tmp_path), "--domain", "cluttered_retrieval", "--num_train", "1",
        "--num_test", "1", "--grasps", "top_down", "--video_hz", "2"
    ],
                   check=True)
    for name in ("train_00", "test_00"):
        info = json.loads((tmp_path / name / "session.json").read_text())
        assert info["domain"] == "cluttered_retrieval"
        boxes = json.loads((tmp_path / name / "gt_boxes.json").read_text())
        for kind in ("target", "obstacle"):
            numbered = sorted(n for n in info["objects"]
                              if n.startswith(kind))
            centres = [boxes[0][n][0] + boxes[0][n][2] for n in numbered]
            assert centres == sorted(centres)
        sim = ClutteredRetrievalMujoco.from_description(info["sim"],
                                                        grasps="top_down")
        try:
            atoms = json.loads((tmp_path / name / "atoms.json").read_text())
            assert sorted([n, list(a)] for n, a in sim.atoms()) == atoms[0]
        finally:
            sim.close()
