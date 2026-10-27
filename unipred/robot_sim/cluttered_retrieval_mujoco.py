"""Cluttered Retrieval in MuJoCo: a stand-in for the real robot of Cluttered
Retrieval Real.

The Franka at the table (unipred/robot_sim/franka_scene.py) is to put snack
bars (the targets) into a black basket (the box) on its right. Obstacles
stand on the table: bottles, which it can pick, and lunch boxes (the bulky
obstacles), which it can only push. All are Google Scanned Objects
(unipred/robot_sim/assets.py), the bottles and lunch boxes drawn by the
seed.

An obstacle blocks a target if, seen from the main camera, it stands in
front of it and hides it: it is nearer the camera, and its image box covers
at least _BLOCK_COVER of the target's. A scene keeps which obstacle blocks
which target, the place of every target (table, box or hand) and which
obstacles were cleared: atoms() are the ground-truth atoms of Cluttered
Retrieval from them, and problems() lists where the simulated scene
disagrees (an obstacle that still hides a target it was cleared from, a
bottle that fell over, a target that missed the basket).

The skills are the robot's scripted motions, with the rules of Cluttered
Retrieval Sim (unipred/envs/cluttered_retrieval_sim.py): PickTarget grasps a
target that no obstacle blocks and goes home with it; PlaceTargetInBox puts
the held target into the basket; RemoveObstacle picks a bottle and stands
it in the corner at the robot's left, behind every target; PushObstacle
lowers the closed gripper beside a lunch box and pushes it sideways, to
where it hides no target. A skill whose conditions do not hold leaves the
scene unchanged.

A scene is built from a Cluttered Retrieval Sim state (from_sim_state): the
targets, and each obstacle with the target it blocks (nearer or farther
from it) or none. Positions are drawn by the seed until the scene's blocks
are the state's and a pushed lunch box would hide no target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from scipy.spatial.transform import Rotation

from predicators.structs import State
from unipred.robot_sim.franka_scene import FINGERTIPS, PADS, \
    AtomTuple, Box2D, FrankaTableScene, MujocoRobot, SimulationError, \
    make_pose, rot_z, top_down
from unipred.robot_sim.franka_scene import mujoco

__all__ = [
    "ClutterLayout", "ClutteredRetrievalMujoco", "MujocoRobot",
    "SimulationError"
]

# Google Scanned Objects (model, scale[, turn]). The targets are snack bars,
# thickened so that the pads close on them above the table; the bottles'
# narrow sides fit between the open fingers (85 mm), those of the bulky
# obstacles (a gift box, a lunch box on its side) do not, and these are low
# enough for the gripper's body to pass over them while its fingertips push
# them.
_TARGET_MODEL = ("Nestle_Candy_19_oz_Butterfinger_Singles_116567",
                 (1.0, 1.1, 2.4))
BOTTLE_MODELS = (
    ("Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original", 1.0),
    ("Marc_Anthony_Skip_Professional_Oil_of_Morocco_Conditioner_with_Argan_Oil",
     1.0),
    ("Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh", 1.0),
)
_ON_SIDE = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)  # a quarter about x
CRATE_MODELS = (
    ("KS_Chocolate_Cube_Box_Assortment_By_Neuhaus_2010_Ounces", 0.55),
    ("Paul_Frank_Dot_Lunch_Box", 0.6, _ON_SIDE),
)
_BASKET_MODEL = ("Curver_Storage_Bin_Black_Small", 1.0)
_TARGET_MASS, _BOTTLE_MASS, _CRATE_MASS = 0.06, 0.25, 0.4
# A bulky obstacle slides on the table (cardboard on a smooth top) rather
# than tipping over when the fingers push it.
_CRATE_FRICTION = 0.4

# The main camera looks at the scene from across the table, low, so that
# obstacles hide the targets behind them.
_CAMERA = ((1.08, 0.10, 0.52), (0.26, 0.03, 0.0), 57.0)
_BLOCK_COVER = 0.2
# Targets lie in a row across the camera's view (lateral y at least
# _TARGET_GAP apart, so that a pushed lunch box hides no other), their long
# sides across it; obstacles stand in front of them, toward the camera (a
# bulky one with its long side toward it), this far from what is in front
# of them: one off the target's centre by up to _FRONT_OFFSET across; of
# two, the second (a bottle) beyond the first and _SECOND_OFFSET to a side,
# so that each hides a part of it.
_TARGET_X = (0.30, 0.36)
_TARGET_Y = (-0.10, 0.56)
_TARGET_GAP = 0.30
_FRONT_GAP = (0.01, 0.03)
_FRONT_OFFSET = 0.03
# How far a bulky obstacle may be pushed sideways: as far as it takes to
# hide nothing.
_PUSHES = np.arange(0.06, 0.205, 0.02)
_SECOND_OFFSET = (0.05, 0.07)
_PUSH_CLEARANCE = 0.05  # of the fingers from the box, coming down
_PUSH_TIPS = 0.02  # the fingertips' height, pushing
# Half sizes (x, y) of the closed gripper's footprint, the fingers closing
# along x: of the fingertips (what a target meets) and of the gripper up to
# 10 cm above them (what anything taller meets).
_FINGERS = np.array([0.035, 0.015])
_GRIPPER = np.array([0.065, 0.045])
# Removed bottles stand here, behind every target as the camera sees it.
_ASIDE = ((0.20, 0.45), (0.20, 0.55), (0.12, 0.40), (0.12, 0.50),
          (0.28, 0.52), (0.12, 0.30))
# The basket: its centre, its long side along x; targets lie in it side by
# side, their long sides along it.
_BASKET_XY = (0.42, -0.36)
_BASKET_SLOTS = (-0.052, 0.0, 0.052)  # y, from its centre
_RELEASE_HEIGHT = 0.02  # of an object's bottom above what it is put on


@dataclass(frozen=True)
class ClutterLayout:
    """The initial scene: the targets (all on the table), and each obstacle
    with its kind ("bottle" or "crate"), the target it blocks ("" if none)
    and its place in front of that target (0 or 1, of two side by side)."""
    targets: Tuple[str, ...]
    obstacles: Tuple[Tuple[str, str, str, int], ...]


class ClutteredRetrievalMujoco(FrankaTableScene):
    """The simulated scene (see module doc)."""
    model_name = "cluttered_retrieval"
    camera = _CAMERA
    # A carried bottle passes over the standing ones.
    carry_z = 0.42

    def __init__(self,
                 layout: ClutterLayout,
                 seed: int = 0,
                 image_size: Tuple[int, int] = (480, 640),
                 grasps: str = "graspgenx") -> None:
        self._layout, self._seed = layout, seed
        self._rng = np.random.default_rng(seed)
        self.targets = list(layout.targets)
        if len(self.targets) > len(_BASKET_SLOTS):
            raise ValueError(f"The basket holds {len(_BASKET_SLOTS)} "
                             "targets.")
        self.kinds = {name: kind for name, kind, _, _ in layout.obstacles}
        bottles = self._rng.permutation(len(BOTTLE_MODELS))
        crates = self._rng.permutation(len(CRATE_MODELS))
        self._models = {t: _TARGET_MODEL for t in self.targets}
        k_bottle = k_crate = 0
        for name, kind, _, _ in layout.obstacles:
            if kind == "bottle":
                self._models[name] = BOTTLE_MODELS[bottles[k_bottle %
                                                           len(bottles)]]
                k_bottle += 1
            else:
                self._models[name] = CRATE_MODELS[crates[k_crate %
                                                         len(crates)]]
                k_crate += 1
        self._models["box"] = _BASKET_MODEL
        self.obstacle_models = {
            n: self._models[n][0]
            for n in self.kinds
        }
        self.place: Dict[str, str] = {t: "table" for t in self.targets}
        self.blocks: Set[Tuple[str, str]] = {
            (name, target)
            for name, _, target, _ in layout.obstacles if target
        }
        self.cleared: Set[str] = set()
        self._pushes: Dict[str, Tuple[float, float]] = {}  # side, distance
        self._slots: Dict[str, float] = {}
        self._aside_used = 0
        super().__init__(image_size, grasps)
        m = self.model
        for g in range(m.ngeom):
            name = m.body(m.geom_bodyid[g]).name
            if self.kinds.get(name) == "crate" and m.geom_group[g] == 3:
                m.geom_friction[g, 0] = _CRATE_FRICTION
        self._reset(layout)

    @classmethod
    def from_sim_state(cls,
                       state: State,
                       seed: int = 0,
                       grasps: str = "graspgenx") -> ClutteredRetrievalMujoco:
        """The scene of a Cluttered Retrieval Sim state: its targets (on the
        table) and obstacles, each with the target it blocks there."""
        targets = sorted(o for o in state if o.type.name == "target")
        if any(state.get(t, "on_table") < 0.5 for t in targets):
            raise ValueError("The simulated scene starts with every target "
                             "on the table.")
        obstacles = []
        for obstacle in sorted(o for o in state if o.type.name == "obstacle"):
            kind = "bottle" if state.get(obstacle, "pickable") > 0.5 else \
                "crate"
            blocked, rank = "", 0
            for target in targets:
                blockers = sorted(
                    (state.get(o, "y"), o.name) for o in state
                    if o.type.name == "obstacle"
                    and _blocks_in_sim(state, o, target))
                names = [name for _, name in blockers]
                if obstacle.name in names:
                    blocked, rank = target.name, names.index(obstacle.name)
            obstacles.append((obstacle.name, kind, blocked, rank))
        return cls(ClutterLayout(tuple(t.name for t in targets),
                                 tuple(obstacles)),
                   seed,
                   grasps=grasps)

    def description(self) -> Dict:
        """What rebuilds this scene as it started (from_description)."""
        return {
            "seed": int(self._seed),
            "targets": list(self._layout.targets),
            "obstacles": [list(o) for o in self._layout.obstacles],
        }

    @classmethod
    def from_description(cls,
                         description: Dict,
                         grasps: str = "graspgenx") -> ClutteredRetrievalMujoco:
        """The scene of a description()."""
        layout = ClutterLayout(
            tuple(description["targets"]),
            tuple((n, k, t, int(r)) for n, k, t, r in description["obstacles"]))
        return cls(layout, description["seed"], grasps=grasps)

    # Observations.

    def atoms(self) -> List[AtomTuple]:
        """The ground-truth atoms of Cluttered Retrieval (without the
        derived Clear)."""
        atoms: List[AtomTuple] = []
        if self.held is None:
            atoms.append(("HandEmpty", ("robot", )))
        for target in self.targets:
            if self.place[target] == "table":
                atoms.append(("OnTable", (target, )))
            elif self.place[target] == "box":
                atoms.append(("InBox", (target, "box")))
            else:
                atoms.append(("Holding", ("robot", target)))
        atoms.extend(("Blocks", pair) for pair in self.blocks)
        for name, kind in self.kinds.items():
            atoms.append(("Pickable" if kind == "bottle" else "Bulky",
                          (name, )))
        return sorted(atoms)

    def objects(self) -> Dict[str, str]:
        """The scene's objects (name -> type)."""
        return {
            "robot": "robot",
            "box": "box",
            **{t: "target"
               for t in self.targets},
            **{o: "obstacle"
               for o in self.kinds},
        }

    def goal(self) -> List[AtomTuple]:
        """The goal of Cluttered Retrieval: every target in the basket."""
        return sorted(("InBox", (t, "box")) for t in self.targets)

    def problems(self) -> List[str]:
        """Where the simulated objects are not where the atoms say."""
        found = []
        basket = self._body_pose("box")
        for target in self.targets:
            centre = self._centre(target)
            local = basket[:3, :3].T @ (centre - basket[:3, 3])
            inside = (abs(local[0]) < self._basket_inner[0]
                      and abs(local[1]) < self._basket_inner[1])
            place = self.place[target]
            if place == "box" and not inside:
                found.append(f"{target} is not in the basket")
            if place == "table" and (inside or centre[2] > 0.08):
                found.append(f"{target} is not on the table")
            if place == "hand" and np.linalg.norm(
                    centre - self._tcp()[:3, 3]) > 0.15:
                found.append(f"{target} is not in the hand")
        for name in self.kinds:
            if self._body_pose(name)[2, 2] < np.cos(np.radians(25)):
                found.append(f"{name} fell over")
        seen = self._seen_blocks()
        for pair in sorted(seen ^ self.blocks):
            found.append(f"{pair[0]} {'hides' if pair in seen else 'does not hide'} "
                         f"{pair[1]}")
        return found

    # Skills.

    def execute(self, option_name: str, objects: Sequence[str]) -> bool:
        """Run a skill of Cluttered Retrieval on the named objects; return
        whether its conditions held (otherwise nothing happens). Raises
        SimulationError if the skill could not be carried out."""
        skills = {
            "PickTarget": self._pick_target,
            "PlaceTargetInBox": self._place_target,
            "RemoveObstacle": self._remove,
            "PushObstacle": self._push,
        }
        if option_name not in skills:
            raise ValueError(f"Unknown skill {option_name}.")
        return skills[option_name](list(objects))

    def _pick_target(self, objects: List[str]) -> bool:
        target = objects[1]
        if self.held is not None or self.place.get(target) != "table" or any(
                t == target for _, t in self.blocks):
            return False
        self._pick(target)
        self.place[target] = "hand"
        return True

    def _place_target(self, objects: List[str]) -> bool:
        target = objects[1]
        if self.held != target:
            return False
        used = set(self._slots.values())
        slot = next(s for s in _BASKET_SLOTS if s not in used)
        basket = self._body_pose("box")
        goal = basket[:3, 3] + basket[:3, :3] @ np.array([0.0, slot, 0.0])
        # Its long side along the basket, its bottom just above the floor.
        self._put_down(goal, basket[2, 3] + self._basket_floor, along=0.0)
        self.place[target] = "box"
        self._slots[target] = slot
        return True

    def _remove(self, objects: List[str]) -> bool:
        obstacle = objects[1]
        if self.held is not None or self.kinds.get(obstacle) != "bottle" or \
                obstacle in self.cleared:
            return False
        self._pick(obstacle)
        x, y = _ASIDE[self._aside_used % len(_ASIDE)]
        self._aside_used += 1
        self._put_down(np.array([x, y, 0.0]), 0.0, along=None)
        self.cleared.add(obstacle)
        self.blocks = {p for p in self.blocks if p[0] != obstacle}
        return True

    def _push(self, objects: List[str]) -> bool:
        obstacle = objects[1]
        if self.held is not None or self.kinds.get(obstacle) != "crate" or \
                obstacle in self.cleared:
            return False
        side, distance = self._pushes[obstacle]
        self._set_fingers(1.0)
        # The closed fingertips just above the table, beside the box, the
        # fingers closing along x (their thin side toward the box).
        start = self._push_start(obstacle, side)
        beside = self._near_home(
            make_pose((start[0], start[1], _PUSH_TIPS), top_down(0.0)))
        beside[:3, 3] -= (PADS + FINGERTIPS) * beside[:3, 2]
        above = beside.copy()
        above[2, 3] += 0.15
        self._move(above)
        self._move(beside)
        end = beside.copy()
        end[1, 3] += side * (distance + _PUSH_CLEARANCE)
        self._move(end)
        back = end.copy()
        back[1, 3] -= side * 0.03
        self._move(back)
        up = back.copy()
        up[2, 3] += 0.15
        self._move(up)
        self._set_fingers(0.0)
        self._go_home()
        self.cleared.add(obstacle)
        self.blocks = {p for p in self.blocks if p[0] != obstacle}
        return True

    # Motions.

    def _put_down(self, goal: np.ndarray, floor: float,
                  along: Optional[float]) -> None:
        """Put the held object down with its footprint's centre over a
        point, its bottom just above a height, turned so that its long side
        runs along a direction (if one; else standing upright), and go
        home."""
        assert self.held is not None
        start = self._site_pose()
        if along is None:
            # Turn the hand so that the object stands upright.
            pose = self._body_pose(self.held)
            axis = np.cross(pose[:3, 2], [0.0, 0.0, 1.0])
            angle = float(np.arccos(np.clip(pose[2, 2], -1.0, 1.0)))
            upright = np.eye(3)
            if np.linalg.norm(axis) > 1e-9:
                upright = Rotation.from_rotvec(
                    axis / np.linalg.norm(axis) * angle).as_matrix()
            target = make_pose(start[:3, 3], upright @ start[:3, :3])
            points = (self._held_points() - start[:3, 3]) @ upright.T
            centre = (points.min(axis=0) + points.max(axis=0)) / 2
            lift = float(points[:, 2].min())
        else:
            target, centre, lift = self._turned(start, along)
        target[:2, 3] = goal[:2] - centre[:2]
        release = target.copy()
        release[2, 3] = floor + _RELEASE_HEIGHT - lift
        target[2, 3] = max(start[2, 3], self.carry_z + PADS)
        self._move(target)
        # Down until it is there, or until what it holds touches something.
        self._move(release, required=False)
        self._detach()
        self._set_fingers(0.0)
        self.held = None
        self._step(250)
        up = self._site_pose()
        up[2, 3] = max(up[2, 3] + 0.12, self.carry_z + PADS)
        self._move(up)
        self._go_home()

    def _centre(self, name: str) -> np.ndarray:
        """The centre of an object's box, in the world."""
        pose = self._body_pose(name)
        points = self._shapes[name].vertices @ pose[:3, :3].T + pose[:3, 3]
        return (points.min(axis=0) + points.max(axis=0)) / 2

    # Blocking.

    def _seen_blocks(self) -> Set[Tuple[str, str]]:
        """The (obstacle, target) pairs where the obstacle hides the target
        from the main camera (see module doc); only targets on the table."""
        boxes = self.boxes()
        camera = self.data.cam_xpos[self.model.camera("main").id]
        seen = set()
        for target in self.targets:
            if self.place[target] != "table":
                continue
            distance = np.linalg.norm(self._centre(target) - camera)
            for obstacle in self.kinds:
                if np.linalg.norm(self._centre(obstacle) - camera) < \
                        distance and _cover(boxes[obstacle],
                                            boxes[target]) >= _BLOCK_COVER:
                    seen.add((obstacle, target))
        return seen

    # The initial scene.

    def _reset(self, layout: ClutterLayout) -> None:
        """Draw positions until the scene's blocks are the layout's and
        every lunch box has a side to be pushed to where it hides
        nothing."""
        self._reset_arm()
        for _ in range(400):
            if self._try_layout(layout):
                mujoco.mj_forward(self.model, self.data)
                self._step(400)
                upright = all(
                    self._body_pose(n)[2, 2] > np.cos(np.radians(10))
                    for n in self.kinds)
                if upright and self._seen_blocks() == self.blocks:
                    return
        raise SimulationError("No layout of the scene shows its blocks.")

    def _try_layout(self, layout: ClutterLayout) -> bool:
        """Place the objects at random positions of the layout; return
        whether they show its blocks and every lunch box can be pushed
        clear."""
        rng = self._rng
        camera_xy = np.array(_CAMERA[0][:2])
        ys = _spaced(len(self.targets), _TARGET_Y, _TARGET_GAP, rng)
        if ys is None:
            return False
        for target, y in zip(self.targets, ys):
            xy = np.array([rng.uniform(*_TARGET_X), y])
            toward = camera_xy - xy
            toward /= np.linalg.norm(toward)
            facing = float(np.arctan2(toward[1], toward[0]))
            across = np.array([-toward[1], toward[0]])
            self._lay(target, xy, facing + np.pi / 2 + rng.uniform(-0.2, 0.2))
            # Its obstacles in front of it: the first about its middle, the
            # second beyond it, to a side. reach is how far what is placed
            # reaches toward the camera.
            reach = float(self._shapes[target].narrow) / 2
            mine = sorted((o for o in layout.obstacles if o[2] == target),
                          key=lambda o: o[3])
            side = rng.choice([-1.0, 1.0])
            for k, (name, _, _, _) in enumerate(mine):
                half = self._half_length(name)
                offset = rng.uniform(-_FRONT_OFFSET, _FRONT_OFFSET) if k == 0 \
                    else side * rng.uniform(*_SECOND_OFFSET)
                along = reach + half + rng.uniform(*_FRONT_GAP)
                centre = xy + toward * along + across * offset
                self._lay(name, centre, facing + rng.uniform(-0.1, 0.1))
                reach = along + half
        # Obstacles that block nothing: in the far corner by the basket,
        # out of the targets' lines of sight.
        free = [o for o in layout.obstacles if not o[2]]
        for k, (name, _, _, _) in enumerate(free):
            self._lay(name, np.array([0.72, -0.44 + 0.22 * k]),
                      rng.uniform(-np.pi, np.pi))
        mujoco.mj_forward(self.model, self.data)
        if self._seen_blocks() != self.blocks or self._overlapping():
            return False
        # Every bulky obstacle: a side to be pushed to, and how far, so that
        # it hides nothing, with room for it and for the fingers.
        for name, kind, _, _ in layout.obstacles:
            if kind == "crate":
                push = self._find_push(name)
                if push is None:
                    return False
                self._pushes[name] = push
        return True

    def _find_push(self, name: str) -> Optional[Tuple[float, float]]:
        """A side (y) and the shortest distance to push a bulky obstacle to
        where it hides no target, with room for it and for the fingers
        (None if there is none)."""
        others = [n for n in self.targets + list(self.kinds) + ["box"]
                  if n != name]
        pose = self._body_pose(name)
        for side in self._rng.permutation([-1.0, 1.0]):
            start = self._push_start(name, side)
            if any(
                    _overlap((start, np.eye(2), _FINGERS if o in self.targets
                              else _GRIPPER), self._footprint(o), 0.01)
                    for o in others):
                continue
            for distance in _PUSHES:
                moved = pose.copy()
                moved[1, 3] += side * distance
                self._set_object_pose(name, moved)
                mujoco.mj_forward(self.model, self.data)
                clear = not any(o == name for o, _ in self._seen_blocks()) \
                    and not self._overlapping(name, grow=0.01)
                self._set_object_pose(name, pose)
                mujoco.mj_forward(self.model, self.data)
                if clear:
                    return float(side), float(distance)
        return None

    def _push_start(self, name: str, side: float) -> np.ndarray:
        """Where the fingers come down to push a lunch box to a side (y)."""
        pose = self._body_pose(name)
        corners = (self._shapes[name].vertices @ pose[:3, :3].T)[:, :2]
        half = float(np.max(np.abs(corners[:, 1] - corners[:, 1].mean())))
        centre = self._centre(name)
        return np.array(
            [centre[0], centre[1] - side * (half + _PUSH_CLEARANCE)])

    def _overlapping(self, name: Optional[str] = None,
                     grow: float = 0.0) -> bool:
        """Whether two objects' footprints (their boxes on the table, turned
        as they stand, 5 mm apart at least) overlap; or, given an object,
        whether its footprint, grown by grow, overlaps another's."""
        names = self.targets + list(self.kinds) + ["box"]
        footprints = {n: self._footprint(n) for n in names}
        pairs = [(a, b) for i, a in enumerate(names) for b in names[:i]]
        if name is not None:
            pairs = [(name, b) for b in names if b != name]
        return any(
            _overlap(footprints[a], footprints[b], 0.005 + grow)
            for a, b in pairs)

    def _footprint(self, name: str) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
        """An object's box on the table: centre, axes (columns) and half
        sizes."""
        shape = self._shapes[name]
        pose = self._body_pose(name)
        centre = pose[:3, :3] @ ((shape.lo + shape.hi) / 2) + pose[:3, 3]
        return centre[:2], pose[:2, :2], (shape.hi - shape.lo)[:2] / 2

    def _half_length(self, name: str) -> float:
        """How far an object reaches from its centre along its long side."""
        return float(self._shapes[name].long) / 2

    def _lay(self, name: str, xy: np.ndarray, yaw: float) -> None:
        """Stand an object on the table, its footprint's centre at xy, its
        long side along yaw."""
        shape = self._shapes[name]
        turn = yaw - (shape.narrow_angle + np.pi / 2)
        centre = rot_z(turn)[:2, :2] @ shape.centre
        self._set_object(name, xy - centre, turn, 0.002)

    # The model.

    def _add_objects(self, spec: mujoco.MjSpec) -> None:
        """The basket, the targets and the obstacles."""
        self._add_basket(spec)
        for name in self.targets:
            self._add_object(spec, name, _TARGET_MASS, convex_parts=False)
        for name, kind in self.kinds.items():
            self._add_object(spec,
                             name,
                             _BOTTLE_MASS if kind == "bottle" else _CRATE_MASS,
                             convex_parts=kind == "bottle")

    def _add_basket(self, spec: mujoco.MjSpec) -> None:
        """The basket where it stands: its mesh (its long side along x) and
        a floor and four walls to collide with."""
        shape = self._shapes["box"]
        half = np.array([(shape.hi[0] - shape.lo[0]) / 2,
                         (shape.hi[1] - shape.lo[1]) / 2])
        height = float(shape.hi[2] - shape.lo[2])
        wall, floor = 0.012, 0.01
        self._basket_inner = half - wall
        self._basket_floor = floor
        body = spec.worldbody.add_body(name="box", pos=[*_BASKET_XY, 0.0])
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
                          group=3)


def _cover(front: Box2D, back: Box2D) -> float:
    """The fraction of the back box that the front box covers."""
    width = max(0.0, min(front[2], back[2]) - max(front[0], back[0]))
    height = max(0.0, min(front[3], back[3]) - max(front[1], back[1]))
    area = (back[2] - back[0]) * (back[3] - back[1])
    return width * height / area if area > 0 else 0.0


def _overlap(a: Tuple[np.ndarray, np.ndarray, np.ndarray],
             b: Tuple[np.ndarray, np.ndarray, np.ndarray],
             margin: float) -> bool:
    """Whether two turned boxes on the table (centre, axes, half sizes),
    each grown by margin / 2, overlap (separating axes)."""
    (ca, ra, ha), (cb, rb, hb) = a, b
    for axis in (*ra.T, *rb.T):
        reach_a = np.sum(np.abs(axis @ ra) * (ha + margin / 2))
        reach_b = np.sum(np.abs(axis @ rb) * (hb + margin / 2))
        if abs(axis @ (cb - ca)) > reach_a + reach_b:
            return False
    return True


def _spaced(count: int, span: Tuple[float, float], gap: float,
            rng: np.random.Generator) -> Optional[List[float]]:
    """count positions in span at least gap apart, in random order (None if
    they do not fit)."""
    slack = span[1] - span[0] - (count - 1) * gap
    if slack < 0:
        return None
    cuts = np.sort(rng.uniform(0, slack, size=count))
    return [
        float(v) for v in rng.permutation(
            [span[0] + cut + i * gap for i, cut in enumerate(cuts)])
    ]


def _blocks_in_sim(state: State, obstacle: object, target: object) -> bool:
    """Whether the obstacle blocks the target in a Cluttered Retrieval Sim
    state."""
    # pylint: disable=import-outside-toplevel
    from unipred.envs.cluttered_retrieval_sim import ClutteredRetrievalSimEnv
    env = ClutteredRetrievalSimEnv
    dx = state.get(obstacle, "x") - state.get(target, "x")
    dy = state.get(obstacle, "y") - state.get(target, "y")
    return state.get(target, "on_table") > 0.5 and 0 < dy <= \
        env.corridor_length and abs(dx) <= env.corridor_half_width
