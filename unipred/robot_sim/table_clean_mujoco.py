"""Table Clean in MuJoCo: a stand-in for the real robot of Table Clean Real.

A Franka Research 3 with a Robotiq 2F-85 gripper (mujoco_menagerie, the
gripper on the flange as on DROID) stands at the edge of a table. On the
table are toys, a wooden tray (the box) on the robot's right, near it or far
from it, a sponge (the wiper) and dirt. The toys, the tray and the sponge
are Google Scanned Objects, the toys drawn by the seed from eight
(unipred/robot_sim/assets.py). A camera across the table renders RGB images.

The seven skills of Table Clean are scripted motions of the arm: straight
lines of the gripper, followed through inverse kinematics by the arm's
position actuators while MuJoCo simulates the scene. A pick takes its grasp
from GraspGenX (grasps="graspgenx": CDG's server, given the object's points
in depth images of the scene; unipred/robot_sim/graspgenx.py) or straight
from above across the object's narrow side (grasps="top_down"). The gripper
closes on the object and a weld then holds it, so that a grasp never slips;
a released object falls and settles. To push or pull the tray, the gripper
presses on its near wall and drags it: the tray slides along x, a drive
keeping it with the gripper. Wiping moves the sponge over the table, and
the dirt goes.

A skill whose conditions do not hold leaves the scene unchanged, with the
rules of Table Clean Sim (unipred/envs/table_clean_sim.py): the robot holds
one object, the sponge leaves and enters the tray only while the tray is
near, and wiping needs the sponge in hand, the tray far and no toy on the
table. The scene keeps the place of every toy and of the sponge (table, box
or hand), the side of the tray and whether the table is clean: atoms() are
the ground-truth atoms of Table Clean from them, and problems() lists where
the simulated objects disagree with them (say, a toy that missed the tray).

A scene is built from a Table Clean Sim state (from_sim_state), so that
Table Clean Sim's tasks and oracle plans can be executed and recorded
(examples/real_robot/collect_sim_demos.py); MujocoRobot drives it through
the closed loop's RobotInterface.
"""

from __future__ import annotations

import functools
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from predicators.structs import State
from unipred.deploy import RobotInterface
from unipred.robot_sim import graspgenx
from unipred.robot_sim.assets import MENAGERIE, SCANNED_OBJECTS, require

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # pylint: disable=wrong-import-position

AtomTuple = Tuple[str, Tuple[str, ...]]
Pose = np.ndarray  # 4 x 4

# The world is the robot's base frame: x forward, y left, z up; the table top
# is z = 0.
_TABLE_X, _TABLE_Y, _TABLE_HEIGHT = (-0.3, 0.85), (-0.65, 0.65), 0.76
_TABLE_RGB = (0.86, 0.85, 0.82)
_TABLE_TEXELS = 1024  # of the table's texture, over its whole top
_DIRT_THICKNESS = 0.0003  # half of the stain layer's, over the table top
_TOY_AREA = ((0.38, 0.62), (0.05, 0.30))  # where toys stand (x, y)
_TOY_GAP = 0.13  # between the centres of toys
_DIRT_AREA = ((0.36, 0.64), (0.03, 0.31))
_TRAY_X = (0.36, 0.60)  # the tray's centre, near and far
_TRAY_Y = -0.23
# Along the tray (y, from its centre): the sponge's place, its long side
# along the tray (so that the open fingers, across it, clear the walls), and
# the toys', their long sides across the tray.
_SPONGE_SLOT = -0.117
_TOY_SLOTS = (0.1405, 0.0628, -0.0149)
_RELEASE_HEIGHT = 0.015  # of an object's bottom above the tray's floor
_CARRY_Z = 0.30  # of the gripper's centre between the pads, carrying
_GRASP_PLANS, _GRASP_TRIES = 3, 6  # GraspGenX plans, grasps tried of each

# Google Scanned Objects (model, scale): toys whose narrow side fits between
# the open fingers (85 mm) and whose long side fits across the tray.
TOY_MODELS = (
    ("Android_Figure_Orange", 1.0),
    ("Android_Figure_Panda", 1.0),
    ("Great_Dinos_Triceratops_Toy", 1.0),  # a green Android figure
    ("Dino_4", 0.8),
    ("FIRE_TRUCK", 0.85),
    ("SCHOOL_BUS", 1.0),
    ("Schleich_Lion_Action_Figure", 1.0),
    ("Toysmith_Windem_Up_Flippin_Animals_Dog", 1.3),
)
_TRAY_MODEL = ("Design_Ideas_Drawer_Store_Organizer", 1.25)
_SPONGE_MODEL = ("Big_O_Sponges_Assorted_Cellulose_12_pack", (0.9, 0.9, 1.6))
_TOY_MASS, _SPONGE_MASS, _TRAY_MASS = 0.08, 0.03, 0.6

# The 2F-85 on DROID's coupling (as CDG's and R2S2R's FR3): its base this far
# beyond link7, turned this far about link7's z; GraspGenX's gripper frame
# ("gripper", z the approach, x the closing direction) on its mounting face;
# the pads' centre, closed, this far along its z, and the fingertips this far
# beyond that.
_FLANGE = 0.107
_GRIPPER_YAW = np.pi / 2
_COUPLING = 0.0108
_PADS = 0.144
_FINGERTIPS = 0.019
_PAD_FRICTION = 1.0
_GRIP_FORCE_LIMIT = 14.4  # the drive's (R2S2R's, as the real 2F-85 squeezes)
_ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
_DROID_HOME = (0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0)
_HOME_TCP = (0.33, 0.0, 0.25)  # the pads' centre at rest, between skills

_TIMESTEP = 0.002
_CONTROL_STEPS = 5  # simulation steps per arm target (100 Hz)
_SPEED = 0.3  # m/s of the gripper
_TURN_SPEED = 1.5  # rad/s of the gripper
_JOINT_SPEED = 1.0  # rad/s of the fastest joint, going home
_CAMERA = ((0.92, 0.30, 0.68), (0.45, -0.06, 0.05), 50.0)  # pos, target, fovy
# Cameras whose depth images make GraspGenX's points: the main one and one
# above the table.
_CLOUD_CAMERAS = ("main", "above")
_CLOUD_SIZE = (240, 320)
_CLOUD_POINTS = 20000  # of the scene around the object


@functools.lru_cache(maxsize=1)
def _table_texels() -> np.ndarray:
    """The table top's texture: its colour with faint speckles (the same in
    every scene)."""
    rng = np.random.default_rng(0)
    texels = np.ones((_TABLE_TEXELS, _TABLE_TEXELS, 3)) * _TABLE_RGB
    texels *= rng.uniform(0.97, 1.0, size=(_TABLE_TEXELS, _TABLE_TEXELS, 1))
    return np.clip(texels * 255, 0, 255).astype(np.uint8)


class SimulationError(Exception):
    """A skill could not be carried out (no reachable grasp, say)."""


@dataclass(frozen=True)
class SceneLayout:
    """The initial scene: toy names with their place ("table", "box" or
    "hand"), the sponge's place, the tray's side (0 near, 1 far) and whether
    the table is clean."""
    toys: Tuple[Tuple[str, str], ...]
    wiper: str
    box_side: int
    clean: bool


@dataclass(frozen=True)
class _Shape:
    """An object's vertices (a sample, in its body frame), its box, and its
    footprint: centre, narrow axis (angle in the body frame) and widths."""
    vertices: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    centre: np.ndarray
    narrow_angle: float
    narrow: float
    long: float


@functools.lru_cache(maxsize=None)
def _shape(model: str, scale: Tuple[float, float, float]) -> _Shape:
    path = require(SCANNED_OBJECTS, f"models/{model}/model.obj")
    with open(path, "r", encoding="utf-8") as f:
        vertices = np.array([[float(v) for v in line.split()[1:4]]
                             for line in f if line.startswith("v ")])
    vertices = vertices * np.array(scale)
    xy = vertices[:, :2]
    _, axes = np.linalg.eigh(np.cov((xy - xy.mean(axis=0)).T))
    narrow_axis, long_axis = axes[:, 0], axes[:, 1]
    across, along = xy @ narrow_axis, xy @ long_axis
    centre = (narrow_axis * (across.min() + across.max()) / 2 + long_axis *
              (along.min() + along.max()) / 2)
    sample = vertices[np.random.default_rng(0).permutation(len(vertices))
                      [:3000]]
    return _Shape(sample, vertices.min(axis=0), vertices.max(axis=0), centre,
                  float(np.arctan2(narrow_axis[1], narrow_axis[0])),
                  float(np.ptp(across)), float(np.ptp(along)))


def _scale3(scale: object) -> Tuple[float, float, float]:
    if isinstance(scale, tuple):
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    return (float(scale), ) * 3  # type: ignore[arg-type]


def _pose(position: Sequence[float], rotation: np.ndarray) -> Pose:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def _top_down(angle: float) -> np.ndarray:
    """GraspGenX's gripper frame pointing down, closing along the angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _yaw_quat(angle: float) -> List[float]:
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


class TableCleanMujoco:
    """The simulated scene (see module doc)."""

    def __init__(self,
                 layout: SceneLayout,
                 seed: int = 0,
                 image_size: Tuple[int, int] = (480, 640),
                 grasps: str = "graspgenx") -> None:
        if grasps not in ("graspgenx", "top_down"):
            raise ValueError(f"Unknown grasps {grasps!r}.")
        self._grasps = grasps
        self._layout, self._seed = layout, seed
        self._rng = np.random.default_rng(seed)
        self.toys = [name for name, _ in layout.toys]
        if len(self.toys) > len(_TOY_SLOTS):
            raise ValueError(f"The tray holds {len(_TOY_SLOTS)} toys.")
        drawn = self._rng.permutation(len(TOY_MODELS))[:len(self.toys)]
        self.toy_models = {t: TOY_MODELS[i][0] for t, i in zip(self.toys, drawn)}
        self._models = {
            **{t: TOY_MODELS[i]
               for t, i in zip(self.toys, drawn)}, "wiper": _SPONGE_MODEL,
            "box": _TRAY_MODEL
        }
        self._shapes = {
            n: _shape(m, _scale3(s))
            for n, (m, s) in self._models.items()
        }
        self.model = self._spec().compile()
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, *image_size)
        self._cloud_renderer: Optional[mujoco.Renderer] = None
        m = self.model
        self._arm_qadr = np.array([m.joint(j).qposadr[0] for j in _ARM_JOINTS])
        self._arm_dof = np.array([m.joint(j).dofadr[0] for j in _ARM_JOINTS])
        self._arm_act = np.array([m.actuator(j).id for j in _ARM_JOINTS])
        self._q_lo = m.jnt_range[[m.joint(j).id for j in _ARM_JOINTS], 0]
        self._q_hi = m.jnt_range[[m.joint(j).id for j in _ARM_JOINTS], 1]
        self._site = m.site("gripper").id
        self._finger_act = m.actuator("gripper/fingers_actuator").id
        self._pads = {
            m.geom(f"gripper/{side}_pad{k}").id: side
            for side in ("left", "right") for k in (1, 2)
        }
        self._tray_act = m.actuator("box_drive").id
        robot_root = m.body("base").id
        self._robot_bodies = set(
            np.flatnonzero(m.body_rootid == m.body_rootid[robot_root]))
        self.place: Dict[str, str] = dict(layout.toys)
        self.place["wiper"] = layout.wiper
        self.box_side = layout.box_side
        self.clean = layout.clean
        self.held: Optional[str] = None
        self._slots: Dict[str, float] = {}
        self._dirt = [m.geom("dirt").id]
        # If a list, the main camera's frames while the simulation runs go
        # to it, one every video_period seconds.
        self.video: Optional[List[np.ndarray]] = None
        self.video_period = 0.2
        self._next_frame = 0.0
        self._q_cmd = np.array(_DROID_HOME)
        self._q_home = self._ik(
            _pose(_HOME_TCP, _top_down(self._home_angle(_DROID_HOME))),
            np.array(_DROID_HOME),
            tcp=True)
        self._home_closing = self._closing_angle(self._q_home)
        # The pads' boxes (centre, half sizes) in the "gripper" site's frame
        # with the gripper open (the kinematics data's fingers are at 0).
        site = self._fk(self._q_home)
        self._open_pads = [
            ((self._ik_data.geom_xpos[g] - site[:3, 3]) @ site[:3, :3],
             np.abs(site[:3, :3].T @ self._ik_data.geom_xmat[g].reshape(3, 3))
             @ m.geom_size[g]) for g in self._pads
        ]
        self._reset(layout)

    @classmethod
    def from_sim_state(cls,
                       state: State,
                       seed: int = 0,
                       grasps: str = "graspgenx") -> TableCleanMujoco:
        """The scene of a Table Clean Sim state (the toys stand where the
        seed puts them)."""
        toys = []
        wiper = "box"
        box_side, clean = 0, False
        for obj in sorted(state, key=lambda o: o.name):
            kind = obj.type.name
            if kind == "toy":
                if state.get(obj, "held") > 0.5:
                    toys.append((obj.name, "hand"))
                elif state.get(obj, "in_box") > 0.5:
                    toys.append((obj.name, "box"))
                else:
                    toys.append((obj.name, "table"))
            elif kind == "wiper":
                wiper = "hand" if state.get(obj, "held") > 0.5 else "box"
            elif kind == "box":
                box_side = int(state.get(obj, "side") > 0.5)
            elif kind == "table":
                clean = bool(state.get(obj, "is_clean") > 0.5)
        return cls(SceneLayout(tuple(toys), wiper, box_side, clean),
                   seed,
                   grasps=grasps)

    def description(self) -> Dict:
        """What rebuilds this scene as it started (from_description)."""
        layout = self._layout
        return {
            "seed": int(self._seed),
            "toys": [list(t) for t in layout.toys],
            "wiper": layout.wiper,
            "box_side": int(layout.box_side),
            "clean": bool(layout.clean),
        }

    @classmethod
    def from_description(cls,
                         description: Dict,
                         grasps: str = "graspgenx") -> TableCleanMujoco:
        """The scene of a description()."""
        layout = SceneLayout(tuple((n, p) for n, p in description["toys"]),
                             description["wiper"], description["box_side"],
                             description["clean"])
        return cls(layout, description["seed"], grasps=grasps)

    def recover(self) -> None:
        """After a skill failed: the arm home, the gripper as it is."""
        self._go_home()

    def close(self) -> None:
        """Free the renderers."""
        self._renderer.close()
        if self._cloud_renderer is not None:
            self._cloud_renderer.close()

    # Observations.

    def render(self, camera: str = "main") -> np.ndarray:
        """An RGB image from a camera."""
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def boxes(self) -> Dict[str, Tuple[float, float, float, float]]:
        """The image box (x0, y0, x1, y1 in pixels) of every object in the
        main camera: its visual geoms' vertices projected, clipped to the
        image (occlusion aside)."""
        m, d = self.model, self.data
        cam = m.camera("main").id
        height, width = self._renderer.height, self._renderer.width
        focal = height / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
        position = d.cam_xpos[cam]
        rotation = d.cam_xmat[cam].reshape(3, 3)
        groups: Dict[str, List[int]] = {"table": [m.geom("table_top").id]}
        for g in range(m.ngeom):
            if m.geom_group[g] > 2 or m.geom_rgba[g, 3] == 0:
                continue
            body = int(m.geom_bodyid[g])
            name = m.body(body).name
            if body in self._robot_bodies:
                groups.setdefault("robot", []).append(g)
            elif name in self.place or name == "box":
                groups.setdefault(name, []).append(g)
        boxes = {}
        for name, geoms in groups.items():
            points = np.concatenate([self._geom_points(g) for g in geoms])
            local = (points - position) @ rotation
            us = width / 2 + focal * local[:, 0] / -local[:, 2]
            vs = height / 2 - focal * local[:, 1] / -local[:, 2]
            boxes[name] = (max(0.0, float(us.min())), max(0.0, float(
                vs.min())), min(float(width), float(us.max())),
                           min(float(height), float(vs.max())))
        return boxes

    def atoms(self) -> List[AtomTuple]:
        """The ground-truth atoms of Table Clean (without the derived
        NoToyOnTable)."""
        atoms: List[AtomTuple] = []
        if self.held is None:
            atoms.append(("HandEmpty", ("robot", )))
        for toy in self.toys:
            if self.place[toy] == "table":
                atoms.append(("ToyOnTable", (toy, "table")))
            elif self.place[toy] == "box":
                atoms.append(("ToyInBox", (toy, "box")))
            else:
                atoms.append(("HoldingToy", ("robot", toy)))
        if self.place["wiper"] == "box":
            atoms.append(("WiperInBox", ("wiper", "box")))
        else:
            atoms.append(("HoldingWiper", ("robot", "wiper")))
        atoms.append(("BoxFar" if self.box_side else "BoxNear", ("box", )))
        if self.clean:
            atoms.append(("TableClean", ("table", )))
        return sorted(atoms)

    def objects(self) -> Dict[str, str]:
        """The scene's objects (name -> type)."""
        return {
            "robot": "robot",
            "table": "table",
            "box": "box",
            "wiper": "wiper",
            **{toy: "toy"
               for toy in self.toys}
        }

    def goal(self) -> List[AtomTuple]:
        """The goal of Table Clean: every toy and the sponge in the tray,
        the table clean."""
        return sorted([("ToyInBox", (toy, "box")) for toy in self.toys] +
                      [("TableClean", ("table", )),
                       ("WiperInBox", ("wiper", "box"))])

    def problems(self) -> List[str]:
        """Where the simulated objects are not where the atoms say."""
        found = []
        tray = self._body_pose("box")
        for name in self.toys + ["wiper"]:
            pose = self._body_pose(name)
            centre = pose[:3, :3] @ np.append(self._shapes[name].centre,
                                              0.0) + pose[:3, 3]
            local = tray[:3, :3].T @ (centre - tray[:3, 3])
            inside = (abs(local[0]) < self._tray_inner[0]
                      and abs(local[1]) < self._tray_inner[1])
            place = self.place[name]
            if place == "box" and not inside:
                found.append(f"{name} is not in the tray")
            if place == "table" and (inside or centre[2] > 0.15):
                found.append(f"{name} is not on the table")
            if place == "hand" and np.linalg.norm(
                    centre - self._tcp()[:3, 3]) > 0.15:
                found.append(f"{name} is not in the hand")
        if abs(tray[0, 3] - _TRAY_X[self.box_side]) > 0.01:
            found.append("the tray is not where its side says")
        dirty = any(self.model.geom_rgba[g, 3] > 0 for g in self._dirt)
        if dirty == self.clean:
            found.append("the dirt does not match TableClean")
        return found

    # Skills.

    def execute(self, option_name: str, objects: Sequence[str]) -> bool:
        """Run a skill of Table Clean on the named objects; return whether
        its conditions held (otherwise nothing happens). Raises
        SimulationError if the skill could not be carried out."""
        skills = {
            "PickToyFromTable": self._pick_toy,
            "PlaceToyInBox": self._place_toy,
            "PickWiperFromBox": self._pick_wiper,
            "PlaceWiperInBox": self._place_wiper,
            "PushBoxOut": self._push_out,
            "PullBoxIn": self._pull_in,
            "WipeTable": self._wipe,
        }
        if option_name not in skills:
            raise ValueError(f"Unknown skill {option_name}.")
        return skills[option_name](list(objects))

    def _pick_toy(self, objects: List[str]) -> bool:
        toy = objects[1]
        if self.held is not None or self.place.get(toy) != "table":
            return False
        self._pick(toy)
        return True

    def _place_toy(self, objects: List[str]) -> bool:
        toy = objects[1]
        if self.held != toy:
            return False
        self._place_in_tray(toy)
        return True

    def _pick_wiper(self, objects: List[str]) -> bool:
        del objects  # the sponge and the tray are the scene's
        if self.held is not None or self.place["wiper"] != "box" or \
                self.box_side != 0:
            return False
        self._pick("wiper")
        return True

    def _place_wiper(self, objects: List[str]) -> bool:
        del objects  # the sponge and the tray are the scene's
        if self.held != "wiper" or self.box_side != 0:
            return False
        self._place_in_tray("wiper")
        return True

    def _push_out(self, objects: List[str]) -> bool:
        del objects  # the tray is the scene's
        if self.box_side != 0:
            return False
        self._drag_tray(1)
        return True

    def _pull_in(self, objects: List[str]) -> bool:
        del objects  # the tray is the scene's
        if self.box_side != 1:
            return False
        self._drag_tray(0)
        return True

    def _wipe(self, objects: List[str]) -> bool:
        del objects  # the sponge, the table and the tray are the scene's
        if self.held != "wiper" or self.box_side != 1 or any(
                self.place[t] == "table" for t in self.toys):
            return False
        # Back and forth along y in rows, the sponge's long side along x, its
        # centre over the dirt, its bottom 2 mm above the table.
        target, centre, lift = self._turned(self._site_pose())
        z = 0.002 - lift
        (x0, x1), (y0, y1) = _DIRT_AREA
        path = []
        for k, x in enumerate(np.linspace(x0, x1, 4)):
            path += [(x, y0), (x, y1)] if k % 2 == 0 else [(x, y1), (x, y0)]
        target[:3, 3] = (path[0][0] - centre[0], path[0][1] - centre[1],
                         z + 0.08)
        self._move(target)
        for x, y in path:
            target[:3, 3] = (x - centre[0], y - centre[1], z)
            self._move(target)
        for g in self._dirt:
            self.model.geom_rgba[g, 3] = 0.0
        self.clean = True
        target = self._site_pose()
        target[2, 3] += 0.15
        self._move(target)
        self._go_home()
        return True

    # Motions.

    def _pick(self, name: str) -> None:
        """Grasp the object (trying grasps in turn, planned anew up to
        _GRASP_PLANS times, until the pads close on it from both sides),
        lift it and go home."""
        tried = 0
        for _ in range(_GRASP_PLANS):
            if any(self._try_grasp(name, grasp)
                   for grasp in self._grasp_candidates(name)[:_GRASP_TRIES]):
                break
            tried += _GRASP_TRIES
        else:
            raise SimulationError(f"No grasp closed on {name} (up to {tried} "
                                  "tried).")
        self._attach(name)
        self.place[name] = "hand"
        self._slots.pop(name, None)
        lift = self._site_pose()
        lift[2, 3] = max(lift[2, 3] + 0.12, _CARRY_Z + _PADS)
        self._move(lift)
        self._go_home()

    def _turned(self,
                start: Pose,
                along: float = 0.0) -> Tuple[Pose, np.ndarray, float]:
        """The hand turned about the vertical so that the held object's long
        side (as it hangs) runs along a direction (an angle from x), the
        object's centre from the hand so turned, and its bottom's height
        from the hand."""
        points = self._held_points()
        xy = points[:, :2] - points[:, :2].mean(axis=0)
        _, axes = np.linalg.eigh(np.cov(xy.T))
        turn = _wrap(along - np.arctan2(axes[1, 1], axes[0, 1]))
        for candidate in (turn, _wrap(turn + np.pi)):
            closing = self._closing_of(_rot_z(candidate) @ start[:3, :3])
            if abs(_wrap(closing - self._home_closing)) <= np.pi / 2 + 1e-6:
                turn = candidate
                break
        rotation = _rot_z(turn)
        turned = (points - start[:3, 3]) @ rotation.T
        centre = (turned.min(axis=0) + turned.max(axis=0)) / 2
        return (_pose(start[:3, 3], rotation @ start[:3, :3]), centre,
                float(turned[:, 2].min()))

    @staticmethod
    def _approach(grasp: Pose) -> Tuple[Pose, Pose, Pose]:
        """The way to a grasp: above it at carrying height, 10 cm back
        along its approach, the grasp."""
        above = grasp.copy()
        above[:3, 3] -= 0.10 * grasp[:3, 2]
        high = above.copy()
        high[2, 3] = max(high[2, 3], _CARRY_Z + _PADS)
        return high, above, grasp

    def _try_grasp(self, name: str, grasp: Pose) -> bool:
        """Close the gripper at a grasp, coming down along its approach;
        return whether the pads hold the object (if not, back off)."""
        high, above, _ = self._approach(grasp)
        self._set_fingers(0.0)
        self._move(high)
        self._move(above)
        if self._move(grasp, required=False):
            self._set_fingers(1.0)
            if self._pads_touch(name):
                return True
            self._set_fingers(0.0)
        self._move(above)
        return False

    def _place_in_tray(self, name: str) -> None:
        if name == "wiper":
            slot = _SPONGE_SLOT
        else:
            used = {s for n, s in self._slots.items() if n != name}
            slot = next(s for s in _TOY_SLOTS if s not in used)
        tray = self._body_pose("box")
        goal = tray[:3, 3] + tray[:3, :3] @ np.array([0.0, slot, 0.0])
        goal += self._rng.uniform(-0.004, 0.004, size=3) * (1, 1, 0)
        start = self._site_pose()
        # A toy's long side across the tray, the sponge's along it, the
        # footprint's centre over the slot, the bottom just above the floor.
        target, centre, lift = self._turned(
            start, np.pi / 2 if name == "wiper" else 0.0)
        target[:2, 3] = goal[:2] - centre[:2]
        floor = tray[2, 3] + self._tray_floor
        release = target.copy()
        release[2, 3] = floor + _RELEASE_HEIGHT - lift
        target[2, 3] = max(start[2, 3], _CARRY_Z + _PADS)
        self._move(target)
        # Down until it is there, or until what it holds touches something.
        self._move(release, required=False)
        self._detach()
        self._set_fingers(0.0)
        self.held = None
        self.place[name] = "box"
        self._slots[name] = slot
        self._step(250)
        up = self._site_pose()
        up[2, 3] = max(up[2, 3] + 0.12, _CARRY_Z + _PADS)
        self._move(up)
        self._go_home()

    def _drag_tray(self, side: int) -> None:
        """Press on the tray's near wall from above and drag it to a side;
        the gripper closed, or holding what it holds."""
        if self.held is None:
            self._set_fingers(1.0)
        start = self._site_pose()
        tray = self._body_pose("box")
        wall_x = tray[0, 3] - self._tray_outer[0] + self._tray_wall / 2
        wall_top = tray[2, 3] + self._tray_height
        if self.held is None:
            contact = start[:3, 3] + start[:3, 2] * (_PADS + _FINGERTIPS)
            lowest = contact[2] - start[2, 3]
        else:
            contact = self._held_centre()
            lowest = self._held_bottom() - start[2, 3]
        offset = start[:3, 3] - contact
        press = start.copy()
        press[:2, 3] = (wall_x + offset[0], tray[1, 3] + offset[1])
        press[2, 3] = wall_top + 0.003 - lowest
        above = press.copy()
        above[2, 3] += 0.10
        self._move(above)
        self._move(press)
        self._tray_follow = tray[0, 3] - press[0, 3]
        end = press.copy()
        end[0, 3] += _TRAY_X[side] - tray[0, 3]
        self._move(end, drag=True)
        self._tray_follow = None
        self.data.ctrl[self._tray_act] = _TRAY_X[side] - _TRAY_X[0]
        self.box_side = side
        up = end.copy()
        up[2, 3] += 0.10
        self._move(up)
        if self.held is None:
            self._set_fingers(0.0)
        self._go_home()

    def _move(self,
              target: Pose,
              drag: bool = False,
              required: bool = True) -> bool:
        """Move the gripper (GraspGenX's frame, the "gripper" site) in a
        straight line to a pose, turning on the way; return whether it got
        within a centimetre (if required, raise SimulationError if not)."""
        start = self._fk(self._q_cmd)
        distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
        turn = float(
            np.linalg.norm(
                Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).as_rotvec()))
        duration = max(distance / _SPEED, turn / _TURN_SPEED, 0.1)
        steps = int(np.ceil(duration / (_TIMESTEP * _CONTROL_STEPS)))
        slerp = Slerp([0.0, 1.0],
                      Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)
            pose = _pose(start[:3, 3] + s * (target[:3, 3] - start[:3, 3]),
                         slerp(s).as_matrix())
            self._q_cmd = self._ik(pose, self._q_cmd)
            self.data.ctrl[self._arm_act] = self._q_cmd
            if drag and self._tray_follow is not None:
                self.data.ctrl[self._tray_act] = (pose[0, 3] +
                                                  self._tray_follow -
                                                  _TRAY_X[0])
            self._step(_CONTROL_STEPS)
        self._settle()
        reached = self._site_pose()
        if np.linalg.norm(reached[:3, 3] - target[:3, 3]) <= 0.01:
            return True
        if required:
            raise SimulationError(
                f"The gripper stopped {reached[:3, 3].round(3)} short of "
                f"{target[:3, 3].round(3)}.")
        # Stop where it is.
        self._q_cmd = self.data.qpos[self._arm_qadr].copy()
        self.data.ctrl[self._arm_act] = self._q_cmd
        return False

    def _go_home(self) -> None:
        """Move the arm to its rest configuration, joint by joint."""
        start = self._q_cmd.copy()
        duration = max(np.max(np.abs(self._q_home - start)) / _JOINT_SPEED,
                       0.1)
        steps = int(np.ceil(duration / (_TIMESTEP * _CONTROL_STEPS)))
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)
            self._q_cmd = start + s * (self._q_home - start)
            self.data.ctrl[self._arm_act] = self._q_cmd
            self._step(_CONTROL_STEPS)
        self._settle()

    def _settle(self, limit: float = 0.6) -> None:
        """Simulate until the arm has caught up with its target."""
        for _ in range(int(limit / (_TIMESTEP * _CONTROL_STEPS))):
            error = self.data.qpos[self._arm_qadr] - self._q_cmd
            if np.max(np.abs(error)) < 2e-3 and np.max(
                    np.abs(self.data.qvel[self._arm_dof])) < 0.02:
                return
            self._step(_CONTROL_STEPS)

    def _set_fingers(self, level: float) -> None:
        """Open (0) or close (1) the gripper and wait for it."""
        self.data.ctrl[self._finger_act] = 255.0 * level
        self._step(300)

    def _step(self, n: int) -> None:
        """Simulate n steps, recording the video (if one) on the way."""
        while n > 0:
            k = n
            if self.video is not None:
                k = min(n, max(1, int(np.ceil(
                    (self._next_frame - self.data.time) / _TIMESTEP))))
            mujoco.mj_step(self.model, self.data, nstep=k)
            n -= k
            if self.video is not None and (self.data.time >=
                                           self._next_frame - 1e-9):
                self.video.append(self.render())
                self._next_frame = self.data.time + self.video_period

    # Grasps.

    def _grasp_candidates(self, name: str) -> List[Pose]:
        """Grasps (poses of the "gripper" site) to pick an object with, best
        first: GraspGenX's from above (within 50 degrees of straight down)
        that the arm reaches, or the one straight down."""
        if self._grasps == "top_down":
            return [self._top_down_grasp(name)]
        points, mask = self._cloud(name)
        poses, scores = graspgenx.plan_grasps(points,
                                              mask,
                                              num_grasps=200,
                                              topk_num_grasps=50,
                                              collision_threshold=0.005)
        surface = points[mask]
        candidates = []
        for pose in poses[np.argsort(-scores)]:
            if pose[2, 2] > -np.cos(np.radians(50)):
                continue
            pose = self._near_home(pose.astype(float))
            if self._pads_clear(pose, surface) and all(
                    self._reaches(p) for p in self._approach(pose)):
                candidates.append(pose)
        return candidates

    def _pads_clear(self, grasp: Pose, surface: np.ndarray) -> bool:
        """Whether the open pads, at a grasp, are clear of the object's
        surface (else they would come down on it)."""
        local = (surface - grasp[:3, 3]) @ grasp[:3, :3]
        for centre, half in self._open_pads:
            if np.any(np.all(np.abs(local - centre) < half, axis=1)):
                return False
        return True

    def _pads_touch(self, name: str) -> bool:
        """Whether both fingers' pads touch the object."""
        m, d = self.model, self.data
        body = m.body(name).id
        sides = set()
        for contact in d.contact[:d.ncon]:
            for pad, other in ((contact.geom1, contact.geom2),
                               (contact.geom2, contact.geom1)):
                if m.geom_bodyid[other] == body and pad in self._pads:
                    sides.add(self._pads[pad])
        return sides == {"left", "right"}

    def _top_down_grasp(self, name: str) -> Pose:
        """Straight down onto the footprint's centre, closing across its
        narrow side, the pads at mid height (25 to 45 mm up)."""
        shape = self._shapes[name]
        pose = self._body_pose(name)
        centre = pose[:3, :3] @ np.append(shape.centre, 0.0) + pose[:3, 3]
        narrow = pose[:3, :3] @ np.array(
            [np.cos(shape.narrow_angle),
             np.sin(shape.narrow_angle), 0.0])
        bottom = float((shape.vertices @ pose[:3, :3].T)[:, 2].min() +
                       pose[2, 3])
        height = shape.hi[2] - shape.lo[2]
        tcp = (centre[0], centre[1],
               bottom + float(np.clip(height / 2, 0.025, 0.045)))
        grasp = _pose(tcp, _top_down(np.arctan2(narrow[1], narrow[0])))
        grasp[:3, 3] -= _PADS * grasp[:3, 2]
        return self._near_home(grasp)

    def _near_home(self, grasp: Pose) -> Pose:
        """The grasp, or the same turned half way about its approach (the
        2F-85 is symmetric), whichever closes nearer the home direction."""
        closing = self._closing_of(grasp[:3, :3])
        if abs(_wrap(closing - self._home_closing)) <= np.pi / 2:
            return grasp
        flipped = grasp.copy()
        flipped[:3, :2] *= -1
        return flipped

    def _reaches(self, pose: Pose) -> bool:
        q = self._ik(pose, self._q_home)
        reached = self._fk(q)
        return bool(np.linalg.norm(reached[:3, 3] - pose[:3, 3]) < 3e-3)

    def _cloud(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """GraspGenX's points: the object's surface (its mesh's vertices, as
        a reconstruction of it would give; the cameras see little of a toy
        at the far end of the table or of the sides of the sponge in the
        tray), and the rest of the scene as depth images of the cloud
        cameras see it, the robot left out; and which are the object's."""
        if self._cloud_renderer is None:
            self._cloud_renderer = mujoco.Renderer(self.model, *_CLOUD_SIZE)
        renderer = self._cloud_renderer
        m, d = self.model, self.data
        height, width = _CLOUD_SIZE
        scene = []
        for camera in _CLOUD_CAMERAS:
            cam = m.camera(camera).id
            renderer.update_scene(d, camera=camera)
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            segments = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            focal = height / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
            v, u = np.mgrid[0:height, 0:width]
            local = np.stack([(u + 0.5 - width / 2) * depth / focal,
                              -(v + 0.5 - height / 2) * depth / focal,
                              -depth], axis=-1).reshape(-1, 3)
            world = local @ d.cam_xmat[cam].reshape(3, 3).T + d.cam_xpos[cam]
            geom = segments[..., 0].reshape(-1)
            is_geom = segments[..., 1].reshape(-1) == int(
                mujoco.mjtObj.mjOBJ_GEOM)
            body = np.where(is_geom & (geom >= 0),
                            m.geom_bodyid[np.maximum(geom, 0)], -1)
            keep = (body >= 0) & (depth.reshape(-1) < 3.0) & ~np.isin(
                body, list(self._robot_bodies) + [m.body(name).id])
            scene.append(world[keep])
        rest = np.concatenate(scene)
        rest = rest[self._rng.permutation(len(rest))[:_CLOUD_POINTS]]
        pose = self._body_pose(name)
        surface = self._shapes[name].vertices @ pose[:3, :3].T + pose[:3, 3]
        points = np.concatenate([surface, rest]).astype(np.float32)
        mask = np.arange(len(points)) < len(surface)
        return points, mask

    # Holding.

    def _attach(self, name: str) -> None:
        """Weld the object to the gripper where it is."""
        m, d = self.model, self.data
        eq = m.equality(f"hold_{name}").id
        hand = m.body("gripper/base").id
        obj = m.body(name).id
        rotation = d.xmat[hand].reshape(3, 3)
        m.eq_data[eq, :3] = 0.0
        m.eq_data[eq, 3:6] = rotation.T @ (d.xpos[obj] - d.xpos[hand])
        relative = np.zeros(4)
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, d.xquat[hand])
        mujoco.mju_mulQuat(relative, inverse, d.xquat[obj])
        m.eq_data[eq, 6:10] = relative
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self.held = name

    def _detach(self) -> None:
        assert self.held is not None
        self.data.eq_active[self.model.equality(f"hold_{self.held}").id] = 0

    def _held_points(self) -> np.ndarray:
        """The held object's vertices (a sample) in the world."""
        assert self.held is not None
        pose = self._body_pose(self.held)
        return self._shapes[self.held].vertices @ pose[:3, :3].T + pose[:3, 3]

    def _held_centre(self) -> np.ndarray:
        points = self._held_points()
        return (points.min(axis=0) + points.max(axis=0)) / 2

    def _held_bottom(self) -> float:
        return float(self._held_points()[:, 2].min())

    # Kinematics.

    def _ik(self, target: Pose, seed: np.ndarray, tcp: bool = False) -> np.ndarray:
        """Arm joints that put the "gripper" site (with tcp, the pads'
        centre) at a pose: damped least squares from the seed, drawn toward
        it in the null space."""
        m, d = self.model, self._ik_data
        q = np.clip(seed, self._q_lo, self._q_hi)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        quat, current, error_quat = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(quat, target[:3, :3].reshape(-1))
        for _ in range(100):
            d.qpos[self._arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rotation = d.site_xmat[self._site].reshape(3, 3)
            position = d.site_xpos[self._site] + (_PADS * rotation[:, 2]
                                                  if tcp else 0.0)
            mujoco.mju_mat2Quat(current, rotation.reshape(-1))
            mujoco.mju_negQuat(current, current)
            mujoco.mju_mulQuat(error_quat, quat, current)
            rotvec = np.zeros(3)
            mujoco.mju_quat2Vel(rotvec, error_quat, 1.0)
            error = np.concatenate([target[:3, 3] - position, rotvec])
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(
                    error[3:]) < 1e-3:
                break
            mujoco.mj_jac(m, d, jacp, jacr, position, m.site_bodyid[self._site])
            jac = np.vstack([jacp[:, self._arm_dof], jacr[:, self._arm_dof]])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6),
                                           error)
            null = np.eye(7) - np.linalg.pinv(jac) @ jac
            step += 0.1 * null @ (seed - q)
            scale = np.max(np.abs(step)) / 0.2
            if scale > 1:
                step /= scale
            q = np.clip(q + step, self._q_lo, self._q_hi)
        return q

    def _fk(self, q: np.ndarray) -> Pose:
        """The "gripper" site's pose at arm joints q."""
        d = self._ik_data
        d.qpos[self._arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return _pose(d.site_xpos[self._site],
                     d.site_xmat[self._site].reshape(3, 3))

    def _site_pose(self) -> Pose:
        """The "gripper" site's pose as simulated."""
        d = self.data
        return _pose(d.site_xpos[self._site].copy(),
                     d.site_xmat[self._site].reshape(3, 3).copy())

    def _tcp(self) -> Pose:
        pose = self._site_pose()
        pose[:3, 3] += _PADS * pose[:3, 2]
        return pose

    def _closing_angle(self, q: np.ndarray) -> float:
        return self._closing_of(self._fk(q)[:3, :3])

    @staticmethod
    def _closing_of(rotation: np.ndarray) -> float:
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _home_angle(self, q: Sequence[float]) -> float:
        return self._closing_angle(np.array(q))

    def _body_pose(self, name: str) -> Pose:
        b = self.model.body(name).id
        return _pose(self.data.xpos[b].copy(),
                     self.data.xmat[b].reshape(3, 3).copy())

    def _geom_points(self, g: int) -> np.ndarray:
        """A geom's vertices (a mesh's, or its box's corners) in the world."""
        m, d = self.model, self.data
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            local = m.mesh_vert[start:start + count]
        else:
            size = m.geom_size[g]
            kind = m.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.array([size[0]] * 3)
            elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER,
                          mujoco.mjtGeom.mjGEOM_CAPSULE):
                half = np.array([size[0], size[0], size[1]])
            else:
                half = np.array(size[:3])
            local = half * np.array(
                np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        return local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    # The initial scene.

    def _reset(self, layout: SceneLayout) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self._arm_qadr] = self._q_home
        d.ctrl[self._arm_act] = self._q_home
        self._q_cmd = self._q_home.copy()
        tray_offset = _TRAY_X[layout.box_side] - _TRAY_X[0]
        d.qpos[m.joint("box_slide").qposadr[0]] = tray_offset
        d.ctrl[self._tray_act] = tray_offset
        self._tray_follow: Optional[float] = None
        mujoco.mj_forward(m, d)
        table_toys = [t for t, place in layout.toys if place == "table"]
        for toy, xy in zip(table_toys, self._toy_positions(len(table_toys))):
            yaw = self._rng.uniform(-np.pi, np.pi)
            self._set_object(toy, xy, yaw, 0.002)
        tray = self._body_pose("box")
        for name in self.toys + ["wiper"]:
            if self.place[name] != "box":
                continue
            slot = _SPONGE_SLOT if name == "wiper" else next(
                s for s in _TOY_SLOTS if s not in self._slots.values())
            self._slots[name] = slot
            shape = self._shapes[name]
            # A toy's long side across the tray, the sponge's along it.
            yaw = (0.0 if name == "wiper" else np.pi / 2) - shape.narrow_angle
            goal = tray[:3, 3] + np.array([0.0, slot, 0.0])
            centre = _rot_z(yaw)[:2, :2] @ shape.centre
            self._set_object(name, goal[:2] - centre, yaw,
                             self._tray_floor + 0.003)
        mujoco.mj_forward(m, d)
        held = [n for n in self.toys + ["wiper"] if self.place[n] == "hand"]
        if held:
            name = held[0]
            # In the hand at home as a pick from above would have it.
            self._set_object(name, (0.0, 0.0), 0.0, 0.0)
            mujoco.mj_forward(m, d)
            grasp = self._top_down_grasp(name)
            home = self._fk(self._q_home)
            relative = home @ np.linalg.inv(grasp)
            pose = relative @ self._body_pose(name)
            self._set_object_pose(name, pose)
            d.ctrl[self._finger_act] = 255.0
            mujoco.mj_forward(m, d)
            self._attach(name)
        self._step(400)

    def _toy_positions(self, count: int) -> List[Tuple[float, float]]:
        (x0, x1), (y0, y1) = _TOY_AREA
        for _ in range(1000):
            xy = self._rng.uniform((x0, y0), (x1, y1), size=(count, 2))
            gaps = [
                np.linalg.norm(xy[i] - xy[j]) for i in range(count)
                for j in range(i)
            ]
            if all(g > _TOY_GAP for g in gaps):
                return [tuple(p) for p in xy]
        raise SimulationError(f"No room on the table for {count} toys.")

    def _set_object(self, name: str, xy: Sequence[float], yaw: float,
                    above: float) -> None:
        """Stand an object upright at (x, y), turned by yaw, its bottom this
        far above the table (or, in the tray, above the table top)."""
        z = above - self._shapes[name].lo[2]
        self._set_object_pose(name, _pose((xy[0], xy[1], z), _rot_z(yaw)))

    def _set_object_pose(self, name: str, pose: Pose) -> None:
        m, d = self.model, self.data
        joint = m.joint(f"{name}_free")
        adr, dof = joint.qposadr[0], joint.dofadr[0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, pose[:3, :3].reshape(-1))
        d.qpos[adr:adr + 3] = pose[:3, 3]
        d.qpos[adr + 3:adr + 7] = quat
        d.qvel[dof:dof + 6] = 0.0

    # The model.

    def _spec(self) -> mujoco.MjSpec:
        """The scene (see module doc)."""
        spec = mujoco.MjSpec.from_file(
            str(require(MENAGERIE, "franka_fr3/fr3.xml")))
        spec.modelname = "table_clean"
        spec.option.timestep = _TIMESTEP
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.visual.quality.shadowsize = 4096
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
        spec.visual.headlight.specular = [0.05, 0.05, 0.05]
        self._add_gripper(spec)
        self._add_room(spec)
        self._add_tray(spec)
        for name in self.toys:
            self._add_object(spec, name, _TOY_MASS, convex_parts=True)
        self._add_object(spec, "wiper", _SPONGE_MASS, convex_parts=False)
        self._add_dirt(spec)
        position, target, fovy = _CAMERA
        self._add_camera(spec, "main", position, target, fovy)
        self._add_camera(spec, "above", (0.45, 0.02, 1.05), (0.47, -0.04, 0.0),
                         60.0)
        return spec

    @staticmethod
    def _add_gripper(spec: mujoco.MjSpec) -> None:
        """The 2F-85 on link7 as on DROID (R2S2R's fr3_robotiq), a site at
        GraspGenX's gripper frame, the arm compensated for gravity."""
        gripper = mujoco.MjSpec.from_file(
            str(require(MENAGERIE, "robotiq_2f85/2f85.xml")))
        for side in ("left", "right"):
            for k in (1, 2):
                gripper.geom(f"{side}_pad{k}").friction[0] = _PAD_FRICTION
        gripper.actuator("fingers_actuator").forcerange = [
            -_GRIP_FORCE_LIMIT, _GRIP_FORCE_LIMIT
        ]
        link7 = spec.body("fr3_link7")
        frame = link7.add_frame(pos=[0.0, 0.0, _FLANGE],
                                quat=_yaw_quat(_GRIPPER_YAW))
        frame.attach_body(gripper.body("base_mount"), "gripper/", "")
        link7.add_site(name="gripper",
                       pos=[0.0, 0.0, _FLANGE + _COUPLING],
                       quat=_yaw_quat(_GRIPPER_YAW),
                       group=5)
        base = spec.body("base")
        for body in [base, *base.find_all(mujoco.mjtObj.mjOBJ_BODY)]:
            body.gravcomp = 1.0

    @staticmethod
    def _add_room(spec: mujoco.MjSpec) -> None:
        """Floor, walls, lights and the table (its top at z = 0)."""
        textures = [
            ("floor", mujoco.mjtBuiltin.mjBUILTIN_CHECKER, [0.46, 0.44, 0.42],
             [0.41, 0.39, 0.37], 0.0),
            ("wall", mujoco.mjtBuiltin.mjBUILTIN_FLAT, [0.78, 0.8, 0.82],
             [0.74, 0.76, 0.78], 0.02),
        ]
        for name, builtin, rgb1, rgb2, random in textures:
            spec.add_texture(name=name,
                             type=mujoco.mjtTexture.mjTEXTURE_2D,
                             builtin=builtin,
                             mark=mujoco.mjtMark.mjMARK_RANDOM,
                             random=random,
                             rgb1=rgb1,
                             rgb2=rgb2,
                             markrgb=rgb2,
                             width=1024,
                             height=1024)
            material = spec.add_material(
                name=name,
                texrepeat=[8, 8] if name == "floor" else [2, 2],
                specular=0.15 if name == "table" else 0.05,
                shininess=0.3)
            material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        texture = spec.add_texture(name="table",
                                   type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   width=_TABLE_TEXELS,
                                   height=_TABLE_TEXELS,
                                   nchannel=3)
        texture.data = _table_texels().tobytes()
        material = spec.add_material(name="table",
                                     texrepeat=[1, 1],
                                     specular=0.15,
                                     shininess=0.3)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table"
        world = spec.worldbody
        world.add_light(name="key",
                        pos=[0.6, 0.6, 2.2],
                        dir=[-0.15, -0.25, -1.0],
                        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        diffuse=[0.45, 0.45, 0.43],
                        castshadow=True)
        world.add_light(name="fill",
                        pos=[1.8, -1.0, 1.5],
                        dir=[-0.6, 0.4, -0.5],
                        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        diffuse=[0.18, 0.18, 0.2],
                        castshadow=False)
        world.add_geom(name="floor",
                       type=mujoco.mjtGeom.mjGEOM_PLANE,
                       size=[4, 4, 0.05],
                       pos=[0, 0, -_TABLE_HEIGHT],
                       material="floor")
        for name, pos, size in [("wall_back", [-1.2, 0, 0.5], [0.02, 3, 1.3]),
                                ("wall_side", [0.5, -1.6, 0.5],
                                 [3, 0.02, 1.3])]:
            world.add_geom(name=name,
                           type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=size,
                           pos=pos,
                           material="wall",
                           contype=0,
                           conaffinity=0)
        sx, sy = np.ptp(_TABLE_X) / 2, np.ptp(_TABLE_Y) / 2
        cx, cy = np.mean(_TABLE_X), np.mean(_TABLE_Y)
        table = world.add_body(name="table", pos=[cx, cy, 0.0])
        table.add_geom(name="table_top",
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[sx, sy, 0.02],
                       pos=[0, 0, -0.02],
                       material="table")
        for i, (lx, ly) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            table.add_geom(
                name=f"table_leg{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.03, 0.03, (_TABLE_HEIGHT - 0.04) / 2],
                pos=[lx * (sx - 0.05), ly * (sy - 0.05),
                     -(_TABLE_HEIGHT + 0.04) / 2],
                rgba=[0.32, 0.33, 0.35, 1.0])

    def _add_textured_mesh(self, spec: mujoco.MjSpec, body: mujoco.MjsBody,
                           name: str, quat: Sequence[float]) -> List[str]:
        """An object's textured visual mesh; returns its collision parts'
        mesh names (added too)."""
        model, scale = self._models[name]
        scale3 = list(_scale3(scale))
        directory = require(SCANNED_OBJECTS, f"models/{model}")
        root = ET.parse(directory / "model.xml").getroot()
        texture = root.find("asset/texture")
        assert texture is not None
        spec.add_texture(name=f"{name}_tex",
                         type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=str(directory / texture.attrib["file"]))
        material = spec.add_material(name=f"{name}_mat",
                                     specular=0.2,
                                     shininess=0.3)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
        spec.add_mesh(name=f"{name}_visual",
                      file=str(directory / "model.obj"),
                      scale=scale3)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"{name}_visual",
                      material=f"{name}_mat",
                      quat=list(quat),
                      contype=0,
                      conaffinity=0,
                      group=2,
                      density=0.0)
        parts = []
        for mesh in root.iter("mesh"):
            if mesh.attrib["name"].startswith("model_collision"):
                part = f"{name}_{mesh.attrib['name']}"
                spec.add_mesh(name=part,
                              file=str(directory / mesh.attrib["file"]),
                              scale=scale3)
                parts.append(part)
        return parts

    def _add_object(self, spec: mujoco.MjSpec, name: str, mass: float,
                    convex_parts: bool) -> None:
        """A free object, held by a weld (off until it is grasped)."""
        body = spec.worldbody.add_body(name=name, pos=[0, 0, 1])
        body.add_freejoint(name=f"{name}_free")
        parts = self._add_textured_mesh(spec, body, name, [1, 0, 0, 0])
        if convex_parts:
            for part in parts:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=part,
                              group=3,
                              mass=mass / len(parts),
                              condim=4)
        else:
            shape = self._shapes[name]
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=((shape.hi - shape.lo) / 2).tolist(),
                          pos=((shape.hi + shape.lo) / 2).tolist(),
                          group=3,
                          mass=mass,
                          condim=4)
        spec.add_equality(name=f"hold_{name}",
                          type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1="gripper/base",
                          name2=name,
                          active=False,
                          solref=[0.01, 1.0])

    def _add_tray(self, spec: mujoco.MjSpec) -> None:
        """The tray: its mesh turned so that its long side runs along y, a
        floor and four walls to collide with, sliding along x on the table,
        a drive holding it where it is put."""
        shape = self._shapes["box"]
        # Turned a quarter, the mesh's x is the tray's y.
        half = np.array([(shape.hi[1] - shape.lo[1]) / 2,
                         (shape.hi[0] - shape.lo[0]) / 2])
        height = float(shape.hi[2] - shape.lo[2])
        wall = 0.01 * _scale3(_TRAY_MODEL[1])[0]
        floor = 0.013 * _scale3(_TRAY_MODEL[1])[2]
        self._tray_outer = half
        self._tray_inner = half - wall
        self._tray_wall = wall
        self._tray_height = height
        self._tray_floor = floor
        body = spec.worldbody.add_body(name="box",
                                       pos=[_TRAY_X[0], _TRAY_Y, 0.0])
        body.add_joint(name="box_slide",
                       type=mujoco.mjtJoint.mjJNT_SLIDE,
                       axis=[1, 0, 0],
                       damping=5.0)
        self._add_textured_mesh(spec, body, "box", _yaw_quat(np.pi / 2))
        boxes = [((half[0], half[1], floor / 2), (0, 0, floor / 2))]
        for sign in (-1, 1):
            boxes.append(((wall / 2, half[1], height / 2),
                          (sign * (half[0] - wall / 2), 0, height / 2)))
            boxes.append(((half[0], wall / 2, height / 2),
                          (0, sign * (half[1] - wall / 2), height / 2)))
        for size, pos in boxes:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=list(size),
                          pos=list(pos),
                          group=3,
                          mass=_TRAY_MASS / len(boxes))
        drive = spec.add_actuator(name="box_drive",
                                  target="box_slide",
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
        drive.set_to_position(kp=3000.0, kv=150.0)

    def _add_dirt(self, spec: mujoco.MjSpec) -> None:
        """Stains on the table: a layer just over the table top whose texture
        is the table's with soft, irregular brown smears painted on (so it
        is seen as stains on the table, not as objects); hidden once the
        table is wiped."""
        texels = _table_texels().astype(np.float32) / 255.0
        size = _TABLE_TEXELS
        rows, cols = np.mgrid[0:size, 0:size]
        # Texel (row, col) -> table (x, y): rows run along y (top row at the
        # largest y), columns along x.
        xs = _TABLE_X[0] + (cols + 0.5) / size * np.ptp(_TABLE_X)
        ys = _TABLE_Y[1] - (rows + 0.5) / size * np.ptp(_TABLE_Y)
        alpha = np.zeros((size, size), dtype=np.float32)
        (x0, x1), (y0, y1) = _DIRT_AREA
        for _ in range(4):
            cx = self._rng.uniform(x0 + 0.04, x1 - 0.04)
            cy = self._rng.uniform(y0 + 0.04, y1 - 0.04)
            strength = self._rng.uniform(0.55, 0.8)
            for _ in range(self._rng.integers(3, 7)):
                bx, by = np.array([cx, cy]) + self._rng.normal(0, 0.015, size=2)
                angle = self._rng.uniform(0, np.pi)
                a, b = self._rng.uniform(0.008, 0.028, size=2)
                c, s_ = np.cos(angle), np.sin(angle)
                u = (xs - bx) * c + (ys - by) * s_
                v = -(xs - bx) * s_ + (ys - by) * c
                blob = np.exp(-0.5 * ((u / a)**2 + (v / b)**2))
                alpha = np.maximum(alpha, strength * blob)
        # A grainy, uneven tint, as dried dirt.
        grain = self._rng.uniform(0.75, 1.0, size=(size, size))
        alpha = (alpha * grain)[..., None]
        stain = np.array([0.45, 0.33, 0.22]) * self._rng.uniform(0.85, 1.1)
        dirty = (texels * (1 - alpha) + stain * alpha) * 255
        texture = spec.add_texture(name="dirt",
                                   type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   width=size,
                                   height=size,
                                   nchannel=3)
        texture.data = np.clip(dirty, 0, 255).astype(np.uint8).tobytes()
        material = spec.add_material(name="dirt",
                                     texrepeat=[1, 1],
                                     specular=0.15,
                                     shininess=0.3)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "dirt"
        spec.worldbody.add_geom(name="dirt",
                                type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[
                                    np.ptp(_TABLE_X) / 2,
                                    np.ptp(_TABLE_Y) / 2, _DIRT_THICKNESS
                                ],
                                pos=[
                                    np.mean(_TABLE_X),
                                    np.mean(_TABLE_Y), _DIRT_THICKNESS
                                ],
                                material="dirt",
                                rgba=[1, 1, 1, 0 if self._layout.clean else 1],
                                contype=0,
                                conaffinity=0,
                                group=1)

    @staticmethod
    def _add_camera(spec: mujoco.MjSpec, name: str, position: Sequence[float],
                    target: Sequence[float], fovy: float) -> None:
        forward = np.subtract(target, position)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat,
                            np.stack([right, up, -forward], axis=1).reshape(-1))
        spec.worldbody.add_camera(name=name,
                                  pos=list(position),
                                  quat=quat.tolist(),
                                  fovy=fovy)


class MujocoRobot(RobotInterface):
    """The closed loop's robot in the simulated scene: a skill that cannot
    be carried out is logged, the arm going home, for the loop to notice.
    Its camera records each skill, a frame every video_period seconds."""

    def __init__(self, sim: TableCleanMujoco, video_period: float = 0.2) -> None:
        self.sim = sim
        self.frames: List[np.ndarray] = []
        self._video: Optional[List[np.ndarray]] = None
        sim.video_period = video_period

    def capture(self) -> np.ndarray:
        image = self.sim.render()
        self.frames.append(image)
        return image

    def execute(self, option_name: str, objects: Sequence[str]) -> None:
        self.sim.video = []
        try:
            if not self.sim.execute(option_name, objects):
                logging.warning(f"{option_name}{tuple(objects)}: its "
                                "conditions do not hold; nothing happened.")
        except SimulationError as e:
            logging.warning(f"{option_name}{tuple(objects)} failed: {e}")
            self.sim.recover()
        finally:
            self._video, self.sim.video = self.sim.video, None

    def video(self) -> Optional[List[np.ndarray]]:
        return self._video
