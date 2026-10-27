"""Table Clean in MuJoCo: a stand-in for the real robot of Table Clean Real.

The Franka at the table (unipred/robot_sim/franka_scene.py) has on the
table toys, a wooden tray (the box) on the robot's right, lengthwise, near
it or pushed out far from it, a sponge (the wiper) and dirt. The toys, the
tray and the sponge are Google Scanned Objects, the toys drawn by the seed
from eight (unipred/robot_sim/assets.py).

The seven skills of Table Clean are the robot's scripted motions. To push
or pull the tray, the gripper presses on its near wall and drags it: the
tray slides along x, a drive keeping it with the gripper. Wiping moves the
sponge over the dirt, which goes.

The rules of Table Clean Sim (unipred/envs/table_clean_sim.py) are the
scene's physics, not checks. The sponge lies at the tray's far end: with
the tray pushed out, it is beyond the arm's reach, so the sponge can be
taken out or put back only while the tray is near (pulled in); the toys go
to the near end, within reach either way. The near tray stands on part of
the dirt, so wiping runs into it until it is pushed out; and the toys
stand on the dirt, so wiping runs into those still on the table. A skill
that cannot be carried out raises SimulationError (where a wipe meets the
tray or a toy, it stops there); only what is impossible outright (picking
with a full hand, placing with an empty one, pushing out a tray already
out) leaves the scene unchanged. The scene keeps the place of every toy and
of the sponge (table, box or hand), the side of the tray and whether the
table is clean: atoms() are the ground-truth atoms of Table Clean from
them, and problems() lists where the simulated objects disagree with them
(say, a toy that missed the tray).

A scene is built from a Table Clean Sim state (from_sim_state), so that
Table Clean Sim's tasks and oracle plans can be executed and recorded
(examples/real_robot/collect_sim_demos.py); MujocoRobot drives it through
the closed loop's RobotInterface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from predicators.structs import State
from unipred.robot_sim.franka_scene import FINGERTIPS, PADS, \
    TABLE_TEXELS, AtomTuple, FrankaTableScene, MujocoRobot, Pose, \
    SimulationError, rot_z, scale3, table_texels
from unipred.robot_sim.franka_scene import mujoco

__all__ = ["MujocoRobot", "SceneLayout", "SimulationError", "TableCleanMujoco"]

_DIRT_THICKNESS = 0.0003  # half of the stain layer's, over the table top

# On the robot's right (y = -0.30) the arm reaches straight down to x = 0.75
# near the table and 0.70 at 30 cm up.
_TRAY_X = (0.42, 0.72)  # the tray's centre, near and pushed out
_TRAY_Y = -0.30
# Along the tray (x, from its centre), the objects' long sides across it:
# the sponge at the far end, with room for the open fingers before the end
# wall (x 0.54 near, 0.84 pushed out: beyond reach), and the toys at the
# near end (at most x 0.68 pushed out: within reach).
_SPONGE_SLOT = 0.12
_TOY_SLOTS = (-0.19, -0.115, -0.04)
# The dirt: under part of the near tray (y below -0.19) and the toys, short
# of the pushed-out tray (from x 0.50); toys stand on it, clear of the
# tray.
_DIRT_AREA = ((0.26, 0.48), (-0.36, 0.28))
_TOY_AREA = ((0.30, 0.46), (-0.08, 0.26))  # where toys stand (x, y)
_TOY_GAP = 0.13  # between the centres of toys
_WIPE_MARGIN = 0.045  # of the sponge's centre from the dirt's edge, along x
_RELEASE_HEIGHT = 0.015  # of an object's bottom above the tray's floor


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
_TRAY_MODEL = ("Design_Ideas_Drawer_Store_Organizer", 1.4)
_SPONGE_MODEL = ("Big_O_Sponges_Assorted_Cellulose_12_pack", (0.9, 0.9, 1.6))
_TOY_MASS, _SPONGE_MASS, _TRAY_MASS = 0.08, 0.03, 0.6

@dataclass(frozen=True)
class SceneLayout:
    """The initial scene: toy names with their place ("table", "box" or
    "hand"), the sponge's place, the tray's side (0 near, 1 far) and whether
    the table is clean."""
    toys: Tuple[Tuple[str, str], ...]
    wiper: str
    box_side: int
    clean: bool


class TableCleanMujoco(FrankaTableScene):
    """The simulated scene (see module doc)."""
    model_name = "table_clean"
    # The table runs out under the pushed-out tray; the camera looks from
    # beyond it.
    table_x = (-0.3, 1.0)
    camera = ((1.12, 0.26, 0.84), (0.42, -0.07, 0.0), 50.0)

    def __init__(self,
                 layout: SceneLayout,
                 seed: int = 0,
                 image_size: Tuple[int, int] = (480, 640),
                 grasps: str = "graspgenx") -> None:
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
        self.place: Dict[str, str] = dict(layout.toys)
        self.place["wiper"] = layout.wiper
        self.box_side = layout.box_side
        self.clean = layout.clean
        self._slots: Dict[str, float] = {}
        super().__init__(image_size, grasps)
        self._tray_act = self.model.actuator("box_drive").id
        self._dirt = [self.model.geom("dirt").id]
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

    # Observations.

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
        self.place[toy] = "hand"
        return True

    def _place_toy(self, objects: List[str]) -> bool:
        toy = objects[1]
        if self.held != toy:
            return False
        self._place_in_tray(toy)
        return True

    def _pick_wiper(self, objects: List[str]) -> bool:
        del objects  # the sponge and the tray are the scene's
        if self.held is not None or self.place["wiper"] != "box":
            return False
        self._pick("wiper")
        self.place["wiper"] = "hand"
        self._slots.pop("wiper", None)
        return True

    def _place_wiper(self, objects: List[str]) -> bool:
        del objects  # the sponge and the tray are the scene's
        if self.held != "wiper":
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
        if self.held != "wiper":
            return False
        # Back and forth along y in rows, the sponge's long side along x, its
        # centre over the dirt, its bottom 2 mm above the table; stopping
        # where it meets the tray or a toy.
        target, centre, lift = self._turned(self._site_pose())
        z = 0.002 - lift
        (x0, x1), (y0, y1) = _DIRT_AREA
        path = []
        for k, x in enumerate(
                np.linspace(x0 + _WIPE_MARGIN, x1 - _WIPE_MARGIN, 4)):
            path += [(x, y0), (x, y1)] if k % 2 == 0 else [(x, y1), (x, y0)]
        target[:3, 3] = (path[0][0] - centre[0], path[0][1] - centre[1],
                         z + 0.08)
        self._move(target)
        for x, y in path:
            target[:3, 3] = (x - centre[0], y - centre[1], z)
            self._move(target, follow=self._wipe_contact)
        for g in self._dirt:
            self.model.geom_rgba[g, 3] = 0.0
        self.clean = True
        target = self._site_pose()
        target[2, 3] += 0.15
        self._move(target)
        self._go_home()
        return True

    def _wipe_contact(self, pose: Pose) -> None:
        """Raise SimulationError if the sponge touches the tray or a toy."""
        del pose  # what matters is what the sponge touches
        m, d = self.model, self.data
        wiper = m.body("wiper").id
        others = {m.body("box").id: "the tray"}
        others.update({
            m.body(t).id: "the tray" if self.place[t] == "box" else t
            for t in self.toys
        })
        for contact in d.contact[:d.ncon]:
            bodies = (m.geom_bodyid[contact.geom1],
                      m.geom_bodyid[contact.geom2])
            for this, other in (bodies, bodies[::-1]):
                if this == wiper and other in others:
                    raise SimulationError(
                        f"Wiping, the sponge ran into {others[other]}.")

    # Motions.

    def _place_in_tray(self, name: str) -> None:
        if name == "wiper":
            slot = _SPONGE_SLOT
        else:
            used = {s for n, s in self._slots.items() if n != name}
            slot = next(s for s in _TOY_SLOTS if s not in used)
        tray = self._body_pose("box")
        goal = tray[:3, 3] + tray[:3, :3] @ np.array([slot, 0.0, 0.0])
        goal += self._rng.uniform(-0.004, 0.004, size=3) * (1, 1, 0)
        start = self._site_pose()
        # Its long side across the tray, the footprint's centre over the
        # slot, the bottom just above the floor.
        target, centre, lift = self._turned(start, np.pi / 2)
        target[:2, 3] = goal[:2] - centre[:2]
        floor = tray[2, 3] + self._tray_floor
        release = target.copy()
        release[2, 3] = floor + _RELEASE_HEIGHT - lift
        target[2, 3] = max(start[2, 3], self.carry_z + PADS)
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
        up[2, 3] = max(up[2, 3] + 0.12, self.carry_z + PADS)
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
            contact = start[:3, 3] + start[:3, 2] * (PADS + FINGERTIPS)
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
        follow = tray[0, 3] - press[0, 3]

        def _drive(pose: Pose) -> None:
            self.data.ctrl[self._tray_act] = pose[0, 3] + follow - _TRAY_X[0]

        end = press.copy()
        end[0, 3] += _TRAY_X[side] - tray[0, 3]
        self._move(end, follow=_drive)
        self.data.ctrl[self._tray_act] = _TRAY_X[side] - _TRAY_X[0]
        self.box_side = side
        up = end.copy()
        up[2, 3] += 0.10
        self._move(up)
        if self.held is None:
            self._set_fingers(0.0)
        self._go_home()

    # The initial scene.

    def _reset(self, layout: SceneLayout) -> None:
        m, d = self.model, self.data
        self._reset_arm()
        tray_offset = _TRAY_X[layout.box_side] - _TRAY_X[0]
        d.qpos[m.joint("box_slide").qposadr[0]] = tray_offset
        d.ctrl[self._tray_act] = tray_offset
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
            # Its long side across the tray (its narrow side along x).
            yaw = -shape.narrow_angle
            goal = tray[:3, 3] + np.array([slot, 0.0, 0.0])
            centre = rot_z(yaw)[:2, :2] @ shape.centre
            self._set_object(name, goal[:2] - centre, yaw,
                             self._tray_floor + 0.003)
        mujoco.mj_forward(m, d)
        held = [n for n in self.toys + ["wiper"] if self.place[n] == "hand"]
        if held:
            self._hold_at_home(held[0])
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

    # The model.

    def _add_objects(self, spec: mujoco.MjSpec) -> None:
        """The tray, the toys, the sponge and the dirt."""
        self._add_tray(spec)
        for name in self.toys:
            self._add_object(spec, name, _TOY_MASS, convex_parts=True)
        self._add_object(spec, "wiper", _SPONGE_MASS, convex_parts=False)
        self._add_dirt(spec)

    def _add_tray(self, spec: mujoco.MjSpec) -> None:
        """The tray: its mesh (its long side along x), a floor and four walls
        to collide with, sliding along x on the table, a drive holding it
        where it is put."""
        shape = self._shapes["box"]
        half = np.array([(shape.hi[0] - shape.lo[0]) / 2,
                         (shape.hi[1] - shape.lo[1]) / 2])
        height = float(shape.hi[2] - shape.lo[2])
        wall = 0.01 * scale3(_TRAY_MODEL[1])[0]
        floor = 0.013 * scale3(_TRAY_MODEL[1])[2]
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
        self._add_textured_mesh(spec, body, "box", [1, 0, 0, 0])
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
        texels = table_texels().astype(np.float32) / 255.0
        size = TABLE_TEXELS
        rows, cols = np.mgrid[0:size, 0:size]
        # Texel (row, col) -> table (x, y): rows run along y (top row at the
        # largest y), columns along x.
        table_x, table_y = self.table_x, self.table_y
        xs = table_x[0] + (cols + 0.5) / size * np.ptp(table_x)
        ys = table_y[1] - (rows + 0.5) / size * np.ptp(table_y)
        alpha = np.zeros((size, size), dtype=np.float32)
        (x0, x1), (y0, y1) = _DIRT_AREA
        # Smears spread over the whole area, from under the near tray to
        # among the toys.
        clusters = 7
        for k in range(clusters):
            cx = self._rng.uniform(x0 + 0.04, x1 - 0.04)
            cy = y0 + 0.04 + (k + self._rng.uniform()) / clusters * (
                y1 - y0 - 0.08)
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
                                    np.ptp(table_x) / 2,
                                    np.ptp(table_y) / 2, _DIRT_THICKNESS
                                ],
                                pos=[
                                    np.mean(table_x),
                                    np.mean(table_y), _DIRT_THICKNESS
                                ],
                                material="dirt",
                                rgba=[1, 1, 1, 0 if self._layout.clean else 1],
                                contype=0,
                                conaffinity=0,
                                group=1)
