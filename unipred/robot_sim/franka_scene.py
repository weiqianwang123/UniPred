"""A Franka at a table in MuJoCo: the robot of the simulated real-robot
domains (unipred/robot_sim/table_clean_mujoco.py,
cluttered_retrieval_mujoco.py).

A Franka Research 3 with a Robotiq 2F-85 gripper (mujoco_menagerie, the
gripper on the flange as on DROID) stands at the edge of a table, a camera
across the table renders RGB images and one above it helps see the scene's
points. A domain's scene adds its objects (Google Scanned Objects,
unipred/robot_sim/assets.py), skills and ground-truth atoms.

Skills are scripted motions of the arm: straight lines of the gripper,
followed through inverse kinematics by the arm's position actuators while
MuJoCo simulates the scene. A pick takes its grasp straight from above
across the object's narrow side (grasps="top_down", the default) or from
GraspGenX (grasps="graspgenx": a GraspGenX server, given the object's
points in depth images of the scene; unipred/robot_sim/graspgenx.py), with
the top-down grasp if none of GraspGenX's closes. The gripper closes on the
object and a weld then holds it, so that a grasp never slips; a released
object falls and settles. While `video` is a list, the main camera's frames
go to it as the simulation runs. MujocoRobot drives a scene through the
closed loop's RobotInterface.
"""

from __future__ import annotations

import functools
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from unipred.deploy import RobotInterface
from unipred.robot_sim import graspgenx
from unipred.robot_sim.assets import MENAGERIE, SCANNED_OBJECTS, require

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # pylint: disable=wrong-import-position

AtomTuple = Tuple[str, Tuple[str, ...]]
Pose = np.ndarray  # 4 x 4
Box2D = Tuple[float, float, float, float]

# The world is the robot's base frame: x forward, y left, z up; the table top
# is z = 0.
TABLE_X, TABLE_Y, TABLE_HEIGHT = (-0.3, 0.85), (-0.65, 0.65), 0.76
TABLE_RGB = (0.86, 0.85, 0.82)
TABLE_TEXELS = 1024  # of the table's texture, over its whole top
CARRY_Z = 0.30  # of the gripper's centre between the pads, carrying
_GRASP_PLANS, _GRASP_TRIES = 3, 6  # GraspGenX plans, grasps tried of each

# The 2F-85 on DROID's coupling (as R2S2R's FR3): its base this far
# beyond link7, turned this far about link7's z; GraspGenX's gripper frame
# ("gripper", z the approach, x the closing direction) on its mounting face;
# the pads' centre, closed, this far along its z, and the fingertips this far
# beyond that.
_FLANGE = 0.107
_GRIPPER_YAW = np.pi / 2
_COUPLING = 0.0108
PADS = 0.144
FINGERTIPS = 0.019
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
def table_texels() -> np.ndarray:
    """The table top's texture: its colour with faint speckles (the same in
    every scene)."""
    rng = np.random.default_rng(0)
    texels = np.ones((TABLE_TEXELS, TABLE_TEXELS, 3)) * TABLE_RGB
    texels *= rng.uniform(0.97, 1.0, size=(TABLE_TEXELS, TABLE_TEXELS, 1))
    return np.clip(texels * 255, 0, 255).astype(np.uint8)


class SimulationError(Exception):
    """A skill could not be carried out (no reachable grasp, say)."""


@dataclass(frozen=True)
class Shape:
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
def object_shape(model: str,
                 scale: Tuple[float, float, float],
                 turn: Optional[Tuple[float, float, float, float]] = None
                 ) -> Shape:
    """The shape of a scanned model, scaled, and turned in its body by a
    quaternion (w, x, y, z) if given (say, to lie a box on its side)."""
    path = require(SCANNED_OBJECTS, f"models/{model}/model.obj")
    with open(path, "r", encoding="utf-8") as f:
        vertices = np.array([[float(v) for v in line.split()[1:4]]
                             for line in f if line.startswith("v ")])
    vertices = vertices * np.array(scale)
    if turn is not None:
        w, x, y, z = turn
        vertices = vertices @ Rotation.from_quat([x, y, z, w]).as_matrix().T
    xy = vertices[:, :2]
    _, axes = np.linalg.eigh(np.cov((xy - xy.mean(axis=0)).T))
    narrow_axis, long_axis = axes[:, 0], axes[:, 1]
    across, along = xy @ narrow_axis, xy @ long_axis
    centre = (narrow_axis * (across.min() + across.max()) / 2 + long_axis *
              (along.min() + along.max()) / 2)
    sample = vertices[np.random.default_rng(0).permutation(len(vertices))
                      [:3000]]
    return Shape(sample, vertices.min(axis=0), vertices.max(axis=0), centre,
                  float(np.arctan2(narrow_axis[1], narrow_axis[0])),
                  float(np.ptp(across)), float(np.ptp(along)))


def scale3(scale: object) -> Tuple[float, float, float]:
    if isinstance(scale, tuple):
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    return (float(scale), ) * 3  # type: ignore[arg-type]


def make_pose(position: Sequence[float], rotation: np.ndarray) -> Pose:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def top_down(angle: float) -> np.ndarray:
    """GraspGenX's gripper frame pointing down, closing along the angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def yaw_quat(angle: float) -> List[float]:
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]

class FrankaTableScene:
    """The robot at the table (see module doc). A domain's scene sets
    self._models (object name -> (scanned model, scale), or (scanned model,
    scale, turn in its body as a quaternion)) and self._rng before calling
    __init__, and adds its objects in _add_objects()."""
    model_name = "franka_table"
    # The main camera (position, target, field of view) and the one above.
    camera = _CAMERA
    above_camera = ((0.45, 0.02, 1.05), (0.47, -0.04, 0.0), 60.0)
    # The height of the pads' centre while carrying, above what stands on
    # the table.
    carry_z = CARRY_Z
    # The table top's extent.
    table_x, table_y = TABLE_X, TABLE_Y

    def __init__(self,
                 image_size: Tuple[int, int] = (480, 640),
                 grasps: str = "top_down") -> None:
        if grasps not in ("graspgenx", "top_down"):
            raise ValueError(f"Unknown grasps {grasps!r}.")
        self._grasps = grasps
        self._models: Dict[str, Tuple]
        self._rng: np.random.Generator
        self._shapes = {
            n: object_shape(m[0], scale3(m[1]), *m[2:])
            for n, m in self._models.items()
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
        robot_root = m.body("base").id
        self._robot_bodies = set(
            np.flatnonzero(m.body_rootid == m.body_rootid[robot_root]))
        self.held: Optional[str] = None
        # If a list, the main camera's frames while the simulation runs go
        # to it, one every video_period seconds.
        self.video: Optional[List[np.ndarray]] = None
        self.video_period = 0.2
        self._next_frame = 0.0
        self._q_cmd = np.array(_DROID_HOME)
        self._q_home = self._ik(
            make_pose(_HOME_TCP, top_down(self._home_angle(_DROID_HOME))),
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

    def objects(self) -> Dict[str, str]:
        """The scene's objects (name -> type)."""
        raise NotImplementedError

    def scanned_models(self) -> Dict[str, str]:
        """The scanned model of each object that has one."""
        return {name: model[0] for name, model in self._models.items()}

    def execute(self, option_name: str, objects: Sequence[str]) -> bool:
        """Run a skill on the named objects; return whether its conditions
        held (otherwise nothing happens). Raises SimulationError if the
        skill could not be carried out."""
        raise NotImplementedError

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

    def boxes(self) -> Dict[str, Box2D]:
        """The image box (x0, y0, x1, y1 in pixels) of every object in the
        main camera: its visual geoms' vertices projected, clipped to the
        image (occlusion aside)."""
        m, d = self.model, self.data
        cam = m.camera("main").id
        height, width = self._renderer.height, self._renderer.width
        focal = height / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
        position = d.cam_xpos[cam]
        rotation = d.cam_xmat[cam].reshape(3, 3)
        named = set(self.objects()) - {"robot", "table"}
        groups: Dict[str, List[int]] = {}
        if "table" in self.objects():
            groups["table"] = [m.geom("table_top").id]
        for g in range(m.ngeom):
            if m.geom_group[g] > 2 or m.geom_rgba[g, 3] == 0:
                continue
            body = int(m.geom_bodyid[g])
            name = m.body(body).name
            if body in self._robot_bodies:
                groups.setdefault("robot", []).append(g)
            elif name in named:
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
    # Motions.

    def _pick(self, name: str) -> None:
        """Grasp the object (trying grasps in turn, planned anew up to
        _GRASP_PLANS times, until the pads close on it from both sides; at
        last straight from above), lift it and go home."""
        tried = 0
        for _ in range(_GRASP_PLANS):
            if any(self._try_grasp(name, grasp)
                   for grasp in self._grasp_candidates(name)[:_GRASP_TRIES]):
                break
            tried += _GRASP_TRIES
        else:
            if self._grasps == "top_down" or not self._try_grasp(
                    name, self._top_down_grasp(name)):
                raise SimulationError(f"No grasp closed on {name} (up to "
                                      f"{tried + 1} tried).")
        self._attach(name)
        lift = self._site_pose()
        lift[2, 3] = max(lift[2, 3] + 0.12, self.carry_z + PADS)
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
        turn = wrap_angle(along - np.arctan2(axes[1, 1], axes[0, 1]))
        for candidate in (turn, wrap_angle(turn + np.pi)):
            closing = self._closing_of(rot_z(candidate) @ start[:3, :3])
            if abs(wrap_angle(closing - self._home_closing)) <= np.pi / 2 + 1e-6:
                turn = candidate
                break
        rotation = rot_z(turn)
        turned = (points - start[:3, 3]) @ rotation.T
        centre = (turned.min(axis=0) + turned.max(axis=0)) / 2
        return (make_pose(start[:3, 3], rotation @ start[:3, :3]), centre,
                float(turned[:, 2].min()))

    def _approach(self, grasp: Pose) -> Tuple[Pose, Pose, Pose]:
        """The way to a grasp: above it at carrying height, 10 cm back
        along its approach, the grasp."""
        above = grasp.copy()
        above[:3, 3] -= 0.10 * grasp[:3, 2]
        high = above.copy()
        high[2, 3] = max(high[2, 3], self.carry_z + PADS)
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
    def _move(self,
              target: Pose,
              required: bool = True,
              follow: Optional[Callable[[Pose], None]] = None) -> bool:
        """Move the gripper (GraspGenX's frame, the "gripper" site) in a
        straight line to a pose, turning on the way (calling follow with
        each pose on the way, say to drive what the gripper drags); return
        whether it got within a centimetre (if required, raise
        SimulationError if not)."""
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
            pose = make_pose(start[:3, 3] + s * (target[:3, 3] - start[:3, 3]),
                         slerp(s).as_matrix())
            self._q_cmd = self._ik(pose, self._q_cmd)
            self.data.ctrl[self._arm_act] = self._q_cmd
            if follow is not None:
                follow(pose)
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
        narrow side, the pads at mid height (25 to 45 mm up), or for a tall
        object 5 cm below its top (so that the gripper's body stays above
        it)."""
        shape = self._shapes[name]
        pose = self._body_pose(name)
        centre = pose[:3, :3] @ np.append(shape.centre, 0.0) + pose[:3, 3]
        narrow = pose[:3, :3] @ np.array(
            [np.cos(shape.narrow_angle),
             np.sin(shape.narrow_angle), 0.0])
        bottom = float((shape.vertices @ pose[:3, :3].T)[:, 2].min() +
                       pose[2, 3])
        height = shape.hi[2] - shape.lo[2]
        tcp = (centre[0], centre[1], bottom + max(
            float(np.clip(height / 2, 0.025, 0.045)), float(height) - 0.05))
        grasp = make_pose(tcp, top_down(np.arctan2(narrow[1], narrow[0])))
        grasp[:3, 3] -= PADS * grasp[:3, 2]
        return self._near_home(grasp)

    def _near_home(self, grasp: Pose) -> Pose:
        """The grasp, or the same turned half way about its approach (the
        2F-85 is symmetric), whichever closes nearer the home direction."""
        closing = self._closing_of(grasp[:3, :3])
        if abs(wrap_angle(closing - self._home_closing)) <= np.pi / 2:
            return grasp
        flipped = grasp.copy()
        flipped[:3, :2] *= -1
        return flipped

    def _reaches(self, pose: Pose) -> bool:
        q = self._ik(pose, self._q_home)
        reached = self._fk(q)
        return bool(np.linalg.norm(reached[:3, 3] - pose[:3, 3]) < 3e-3)

    def _seen_points(self, camera: str) -> Tuple[np.ndarray, np.ndarray]:
        """The points of the scene that a camera sees (its depth image, at
        the cloud resolution, within 3 m) and the body of each."""
        if self._cloud_renderer is None:
            self._cloud_renderer = mujoco.Renderer(self.model, *_CLOUD_SIZE)
        renderer = self._cloud_renderer
        m, d = self.model, self.data
        height, width = _CLOUD_SIZE
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
                          -(v + 0.5 - height / 2) * depth / focal, -depth],
                         axis=-1).reshape(-1, 3)
        world = local @ d.cam_xmat[cam].reshape(3, 3).T + d.cam_xpos[cam]
        geom = segments[..., 0].reshape(-1)
        is_geom = segments[..., 1].reshape(-1) == int(mujoco.mjtObj.mjOBJ_GEOM)
        body = np.where(is_geom & (geom >= 0),
                        m.geom_bodyid[np.maximum(geom, 0)], -1)
        keep = (body >= 0) & (depth.reshape(-1) < 3.0)
        return world[keep], body[keep]

    def _cloud(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """GraspGenX's points: the object's surface (its mesh's vertices, as
        a reconstruction of it would give; the cameras see little of an object
        at the far end of the table or of the sides of the sponge in the
        tray), and the rest of the scene as depth images of the cloud
        cameras see it, the robot left out; and which are the object's."""
        scene = []
        for camera in _CLOUD_CAMERAS:
            world, body = self._seen_points(camera)
            keep = ~np.isin(body,
                            list(self._robot_bodies) + [self.model.body(name).id])
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
            position = d.site_xpos[self._site] + (PADS * rotation[:, 2]
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
        return make_pose(d.site_xpos[self._site],
                     d.site_xmat[self._site].reshape(3, 3))

    def _site_pose(self) -> Pose:
        """The "gripper" site's pose as simulated."""
        d = self.data
        return make_pose(d.site_xpos[self._site].copy(),
                     d.site_xmat[self._site].reshape(3, 3).copy())

    def _tcp(self) -> Pose:
        pose = self._site_pose()
        pose[:3, 3] += PADS * pose[:3, 2]
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
        return make_pose(self.data.xpos[b].copy(),
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

    def _reset_arm(self) -> None:
        """The simulation reset, the arm at home."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self._arm_qadr] = self._q_home
        d.ctrl[self._arm_act] = self._q_home
        self._q_cmd = self._q_home.copy()
        mujoco.mj_forward(m, d)

    def _hold_at_home(self, name: str) -> None:
        """The object in the hand at home, as a pick from above would have
        it."""
        m, d = self.model, self.data
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

    def _set_object(self, name: str, xy: Sequence[float], yaw: float,
                    above: float) -> None:
        """Stand an object upright at (x, y), turned by yaw, its bottom this
        far above the table (or, in the tray, above the table top)."""
        z = above - self._shapes[name].lo[2]
        self._set_object_pose(name, make_pose((xy[0], xy[1], z), rot_z(yaw)))

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
        """The robot, the room, the domain's objects and the cameras."""
        spec = mujoco.MjSpec.from_file(
            str(require(MENAGERIE, "franka_fr3/fr3.xml")))
        spec.modelname = self.model_name
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
        self._add_objects(spec)
        self._add_camera(spec, "main", *self.camera)
        self._add_camera(spec, "above", *self.above_camera)
        return spec

    def _add_objects(self, spec: mujoco.MjSpec) -> None:
        """The domain's objects."""
        raise NotImplementedError

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
                                quat=yaw_quat(_GRIPPER_YAW))
        frame.attach_body(gripper.body("base_mount"), "gripper/", "")
        link7.add_site(name="gripper",
                       pos=[0.0, 0.0, _FLANGE + _COUPLING],
                       quat=yaw_quat(_GRIPPER_YAW),
                       group=5)
        base = spec.body("base")
        for body in [base, *base.find_all(mujoco.mjtObj.mjOBJ_BODY)]:
            body.gravcomp = 1.0

    def _add_room(self, spec: mujoco.MjSpec) -> None:
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
                                   width=TABLE_TEXELS,
                                   height=TABLE_TEXELS,
                                   nchannel=3)
        texture.data = table_texels().tobytes()
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
                       pos=[0, 0, -TABLE_HEIGHT],
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
        sx, sy = np.ptp(self.table_x) / 2, np.ptp(self.table_y) / 2
        cx, cy = np.mean(self.table_x), np.mean(self.table_y)
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
                size=[0.03, 0.03, (TABLE_HEIGHT - 0.04) / 2],
                pos=[lx * (sx - 0.05), ly * (sy - 0.05),
                     -(TABLE_HEIGHT + 0.04) / 2],
                rgba=[0.32, 0.33, 0.35, 1.0])

    def _add_textured_mesh(self, spec: mujoco.MjSpec, body: mujoco.MjsBody,
                           name: str, quat: Sequence[float]) -> List[str]:
        """An object's textured visual mesh, turned by quat (after the
        model's own turn, if any); returns its collision parts' mesh names
        (added too)."""
        model, scale = self._models[name][:2]
        scales = list(scale3(scale))
        turn = self._model_turn(name)
        if turn is not None:
            turned = np.zeros(4)
            mujoco.mju_mulQuat(turned, np.asarray(quat, float),
                               np.asarray(turn, float))
            quat = turned.tolist()
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
                      scale=scales)
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
                              scale=scales)
                parts.append(part)
        return parts

    def _model_turn(
            self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """The turn of an object's model in its body, if any."""
        return self._models[name][2] if len(self._models[name]) > 2 else None

    def _add_object(self, spec: mujoco.MjSpec, name: str, mass: float,
                    convex_parts: bool) -> None:
        """A free object, held by a weld (off until it is grasped)."""
        body = spec.worldbody.add_body(name=name, pos=[0, 0, 1])
        body.add_freejoint(name=f"{name}_free")
        parts = self._add_textured_mesh(spec, body, name, [1, 0, 0, 0])
        if convex_parts:
            turn = self._model_turn(name)
            for part in parts:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=part,
                              quat=list(turn or (1.0, 0.0, 0.0, 0.0)),
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

    def __init__(self,
                 sim: FrankaTableScene,
                 video_period: float = 0.2) -> None:
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
