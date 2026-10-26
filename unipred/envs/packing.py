"""Packing: put every item into a box, adjusting the boxes' dividers so the
items fit.

A box is split by a movable divider into a left and a right region, and
each region holds at most one item. Items are small or large. With the
divider in the middle, both regions of a box fit a small item but not a
large one; shifting the divider makes one region wide enough for a large
item and leaves the other region still wide enough for a small one. The
divider can only be shifted while both regions are empty, and only once.

The state of an object is its pose, a few robot-visible flags and the
feature vector of its top-down image crop, computed by a frozen image
encoder (unipred/perception/features.py: DINOv3, or ResNet-18 if DINOv3 is
not installed). How wide a region is, whether it is occupied and how large
an item is are not features: they are only visible in the crops, so the
predicates about them are learned from the image features. The simulator
keeps them in State.simulator_state, which the ground-truth predicates
read.

Options: Pick(robot, item) from the table; PlaceSmall(robot, item, region)
and PlaceLarge(robot, item, region), the placing skills for small and large
items, whose parameters choose the spot in the region; Shift(robot, region,
other) shifts the divider between two regions of a box so that the first
becomes wide, its parameter choosing how far.

Train tasks have two boxes and three items (one large); test tasks have
three boxes and four items (two large). The goal is every item packed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from PIL import Image, ImageDraw

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type
from unipred.perception.features import CropEncoder

# Primitives, the first entry of an action [primitive, a, b, c]: pick at
# (a, b); place at (a, b); set the divider of the box at (a, b) to the
# fraction c of its width.
PICK, PLACE, SHIFT = range(3)


@dataclass
class PackingScene:
    """What only the simulator knows (State.simulator_state)."""
    # Item name -> "small" or "large".
    item_sizes: Dict[str, str]
    # Box index -> divider position, as a fraction of the box width from
    # its left edge (0.5: centered).
    dividers: Dict[int, float]
    # Region name -> (box index, 0 for left or 1 for right).
    regions: Dict[str, Tuple[int, int]]
    # Region name -> name of the item in it ("" if empty).
    contents: Dict[str, str]


class PackingEnv(BaseEnv):
    """Packing (see module doc)."""
    # Boxes stand in a row at y = box_y; items start in the strip below.
    box_y: ClassVar[float] = 0.75
    box_width: ClassVar[float] = 0.26
    box_height: ClassVar[float] = 0.16
    box_spacing: ClassVar[float] = 0.32
    item_sizes: ClassVar[Dict[str, float]] = {"small": 0.08, "large": 0.14}
    table_y: ClassVar[Tuple[float, float]] = (0.08, 0.4)
    min_dist: ClassVar[float] = 0.18
    # A shift sets the divider to 0.5 +- 0.5 * param (toward the narrow
    # region); a large item fits the wide region for params in about
    # [0.16, 0.35].
    max_shift: ClassVar[float] = 0.5
    reach: ClassVar[float] = 0.03
    # Side of an object's square crop in world units and pixels.
    crop_size: ClassVar[float] = 0.3
    crop_pixels: ClassVar[int] = 64
    colors: ClassVar[Dict[str, Tuple[int, int, int]]] = {
        "background": (230, 230, 230),
        "box": (70, 70, 70),
        "divider": (200, 40, 40),
        "small": (40, 90, 200),
        "large": (40, 160, 70),
    }

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._encoder = CropEncoder(CFG.perception_encoder, CFG.device)
        crop_features = [f"f{i}" for i in range(self._encoder.dim)]
        self.robot_type = Type("robot", ["x", "y", "gripper"])
        self.item_type = Type("item", ["x", "y", "held"] + crop_features)
        self.region_type = Type("region", ["x", "y"] + crop_features)
        robot, item, region = self.robot_type, self.item_type, \
            self.region_type
        self._HandEmpty = Predicate("HandEmpty", [robot], self._hand_empty)
        self._Holding = Predicate("Holding", [robot, item], self._holding)
        self._OnTable = Predicate("OnTable", [item], self._on_table)
        self._Packed = Predicate("Packed", [item], self._packed)
        self._In = Predicate("In", [item, region], self._in)
        self._Wide = Predicate("Wide", [region], self._wide)
        self._Centered = Predicate("Centered", [region], self._centered)
        self._Free = Predicate("Free", [region], self._free)
        self._IsSmall = Predicate("IsSmall", [item],
                                  self._size_is("small"))
        self._IsLarge = Predicate("IsLarge", [item],
                                  self._size_is("large"))
        self._SameBox = Predicate("SameBox", [region, region],
                                  self._same_box)

    @classmethod
    def get_name(cls) -> str:
        return "packing"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._Holding, self._OnTable, self._Packed,
            self._In, self._Wide, self._Centered, self._Free, self._IsSmall,
            self._IsLarge, self._SameBox
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Packed}

    @property
    def derived_predicates(self) -> Set[Predicate]:
        return {self._Free}

    @property
    def types(self) -> Set[Type]:
        return {self.robot_type, self.item_type, self.region_type}

    @property
    def action_space(self) -> Box:
        return Box(np.array([0, 0, 0, 0], dtype=np.float32),
                   np.array([SHIFT, 1, 1, 1], dtype=np.float32))

    # Simulation.

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        primitive = int(round(float(action.arr[0])))
        x, y, fraction = (float(v) for v in action.arr[1:])
        scene: PackingScene = state.simulator_state
        robot = state.get_objects(self.robot_type)[0]
        held = self._held_item(state)
        next_state = state.copy()
        next_scene: PackingScene = next_state.simulator_state
        if primitive == PICK:
            item = self._item_on_table_at(state, x, y)
            if held is not None or item is None:
                return next_state
            next_state.set(item, "x", x)
            next_state.set(item, "y", y)
            next_state.set(item, "held", 1.0)
            next_state.set(robot, "gripper", 1.0)
        elif primitive == PLACE:
            region = self._region_at(state, x)
            if held is None or region is None or scene.contents[region.name]:
                return next_state
            left, right = self.region_bounds(scene, region.name)
            half = self.item_sizes[scene.item_sizes[held.name]] / 2
            if x - half < left or x + half > right:
                return next_state  # does not fit
            next_state.set(held, "x", x)
            next_state.set(held, "y", self.box_y)
            next_state.set(held, "held", 0.0)
            next_state.set(robot, "gripper", 0.0)
            next_scene.contents[region.name] = held.name
        elif primitive == SHIFT:
            box = self._box_at(scene, x, y)
            if box is None or abs(scene.dividers[box] - 0.5) > 1e-6 or any(
                    scene.contents[name]
                    for name, (b, _) in scene.regions.items() if b == box):
                return next_state
            next_scene.dividers[box] = fraction
        else:
            return next_state
        next_state.set(robot, "x", x)
        next_state.set(robot, "y", y)
        self._update_crops(next_state)
        return next_state

    def _update_crops(self, state: State) -> None:
        """Render and encode the crops of the items and regions."""
        objects = [
            obj for obj in state if obj.type in (self.item_type,
                                                 self.region_type)
        ]
        crops = [self.render_crop(state, obj) for obj in objects]
        for obj, features in zip(objects, self._encoder.encode(crops)):
            data = state.data[obj]
            data[len(obj.type.feature_names) - len(features):] = features

    # Tasks.

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [
            self._make_task(CFG.packing_num_boxes_train,
                            CFG.packing_num_large_train,
                            CFG.packing_num_small_train, self._train_rng)
            for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            sizes = (CFG.packing_num_boxes_train, CFG.packing_num_large_train,
                     CFG.packing_num_small_train)
        else:
            sizes = (CFG.packing_num_boxes_test, CFG.packing_num_large_test,
                     CFG.packing_num_small_test)
        return [
            self._make_task(*sizes, self._test_rng)
            for _ in range(CFG.num_test_tasks)
        ]

    def _make_task(self, num_boxes: int, num_large: int, num_small: int,
                   rng: np.random.Generator) -> EnvironmentTask:
        assert num_large <= num_boxes and \
            num_large + num_small <= 2 * num_boxes, "The items do not fit."
        robot = Object("robot", self.robot_type)
        sizes = ["large"] * num_large + ["small"] * num_small
        rng.shuffle(sizes)
        items = [Object(f"item{i}", self.item_type) for i in range(len(sizes))]
        regions = [
            Object(f"box{b}_{side}", self.region_type)
            for b in range(num_boxes) for side in ("left", "right")
        ]
        scene = PackingScene(
            item_sizes={item.name: size
                        for item, size in zip(items, sizes)},
            dividers={b: 0.5
                      for b in range(num_boxes)},
            regions={
                region.name: (i // 2, i % 2)
                for i, region in enumerate(regions)
            },
            contents={region.name: ""
                      for region in regions})
        dim = self._encoder.dim
        data: Dict[Object, np.ndarray] = {
            robot: np.array([0.5, 0.0, 0.0], dtype=np.float32)
        }
        for item, (x, y) in zip(items, self._item_spots(len(items), rng)):
            data[item] = np.concatenate([[x, y, 0.0], np.zeros(dim)])
        for region in regions:
            box, side = scene.regions[region.name]
            x = self.box_x(box) + (side - 0.5) * self.box_width / 2
            data[region] = np.concatenate([[x, self.box_y], np.zeros(dim)])
        state = State({obj: np.asarray(v, dtype=np.float32)
                       for obj, v in data.items()},
                      simulator_state=scene)
        self._update_crops(state)
        goal = {GroundAtom(self._Packed, [item]) for item in items}
        return EnvironmentTask(state, goal)

    def _item_spots(self, num: int,
                    rng: np.random.Generator) -> List[Tuple[float, float]]:
        """Rejection-sample item positions on the table, far apart."""
        spots: List[Tuple[float, float]] = []
        while len(spots) < num:
            spot = (float(rng.uniform(0.1, 0.9)),
                    float(rng.uniform(*self.table_y)))
            if all(
                    np.hypot(spot[0] - s[0], spot[1] - s[1]) >= self.min_dist
                    for s in spots):
                spots.append(spot)
        return spots

    # Rendering.

    def render_crop(self, state: State, obj: Object) -> np.ndarray:
        """The top-down image crop of an item or a region, centered at its
        pose (crop_size world units, crop_pixels pixels)."""
        scale = self.crop_pixels / self.crop_size
        cx, cy = state.get(obj, "x"), state.get(obj, "y")

        def _px(x: float, y: float) -> Tuple[float, float]:
            return ((x - cx) * scale + self.crop_pixels / 2,
                    (cy - y) * scale + self.crop_pixels / 2)

        image = Image.new("RGB", (self.crop_pixels, self.crop_pixels),
                          self.colors["background"])
        draw = ImageDraw.Draw(image)
        self._draw_scene(state, draw, _px, scale)
        return np.asarray(image)

    def _draw_scene(self, state: State, draw: ImageDraw.ImageDraw,
                    to_px: Callable[[float, float], Tuple[float, float]],
                    scale: float) -> None:
        """Draw the boxes, dividers and items."""
        scene: PackingScene = state.simulator_state
        for box, fraction in scene.dividers.items():
            left = self.box_x(box) - self.box_width / 2
            x0, y0 = to_px(left, self.box_y + self.box_height / 2)
            x1, y1 = to_px(left + self.box_width,
                           self.box_y - self.box_height / 2)
            draw.rectangle([x0, y0, x1, y1],
                           outline=self.colors["box"],
                           width=max(1, int(0.008 * scale)))
            dx, _ = to_px(left + fraction * self.box_width, self.box_y)
            draw.line([dx, y0, dx, y1],
                      fill=self.colors["divider"],
                      width=max(1, int(0.008 * scale)))
        for item in state.get_objects(self.item_type):
            half = self.item_sizes[scene.item_sizes[item.name]] / 2
            x, y = state.get(item, "x"), state.get(item, "y")
            x0, y0 = to_px(x - half, y + half)
            x1, y1 = to_px(x + half, y - half)
            draw.rectangle([x0, y0, x1, y1],
                           fill=self.colors[scene.item_sizes[item.name]])

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        pixels = 400
        image = Image.new("RGB", (pixels, pixels), self.colors["background"])
        self._draw_scene(state, ImageDraw.Draw(image), lambda x, y:
                         (x * pixels, (1 - y) * pixels), pixels)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(np.asarray(image), extent=(0, 1, 0, 1))
        for obj in state.get_objects(self.item_type):
            ax.annotate(obj.name, (state.get(obj, "x"), state.get(obj, "y")))
        title = f"goal: {sorted(task.goal)}"
        if action is not None:
            title += f"\naction: {action.arr}"
        if caption is not None:
            title += f"\n{caption}"
        ax.set_title(title, fontsize=6)
        ax.axis("off")
        plt.tight_layout()
        return fig

    # Geometry.

    @classmethod
    def box_x(cls, box: int) -> float:
        """The x of the center of a box."""
        return 0.5 + (box - 1) * cls.box_spacing

    @classmethod
    def region_bounds(cls, scene: PackingScene,
                      region_name: str) -> Tuple[float, float]:
        """The left and right x of a region (they depend on the divider)."""
        box, side = scene.regions[region_name]
        left = cls.box_x(box) - cls.box_width / 2
        divider = left + scene.dividers[box] * cls.box_width
        if side == 0:
            return left, divider
        return divider, left + cls.box_width

    def _region_at(self, state: State, x: float) -> Optional[Object]:
        """The region whose current bounds contain x."""
        scene: PackingScene = state.simulator_state
        for region in state.get_objects(self.region_type):
            left, right = self.region_bounds(scene, region.name)
            if left <= x <= right:
                return region
        return None

    def _box_at(self, scene: PackingScene, x: float,
                y: float) -> Optional[int]:
        for box in scene.dividers:
            if abs(x - self.box_x(box)) <= self.reach and abs(
                    y - self.box_y) <= self.reach:
                return box
        return None

    def _held_item(self, state: State) -> Optional[Object]:
        for item in state.get_objects(self.item_type):
            if state.get(item, "held") > 0.5:
                return item
        return None

    def _item_on_table_at(self, state: State, x: float,
                          y: float) -> Optional[Object]:
        for item in state.get_objects(self.item_type):
            if self._on_table(state, [item]) and np.hypot(
                    state.get(item, "x") - x,
                    state.get(item, "y") - y) <= self.reach:
                return item
        return None

    def region_width(self, state: State, region: Object) -> float:
        """The current width of a region."""
        left, right = self.region_bounds(state.simulator_state, region.name)
        return right - left

    # Predicate classifiers (ground truth: they may read the scene).

    @staticmethod
    def _hand_empty(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "gripper") < 0.5

    @staticmethod
    def _holding(state: State, objects: Sequence[Object]) -> bool:
        _, item = objects
        return state.get(item, "held") > 0.5

    @staticmethod
    def _packed(state: State, objects: Sequence[Object]) -> bool:
        item, = objects
        return item.name in state.simulator_state.contents.values()

    def _on_table(self, state: State, objects: Sequence[Object]) -> bool:
        item, = objects
        return not self._holding(state, [None, item]) and \
            not self._packed(state, [item])

    @staticmethod
    def _in(state: State, objects: Sequence[Object]) -> bool:
        item, region = objects
        return state.simulator_state.contents[region.name] == item.name

    def _wide(self, state: State, objects: Sequence[Object]) -> bool:
        region, = objects
        return self.region_width(state, region) >= self.item_sizes["large"]

    @staticmethod
    def _centered(state: State, objects: Sequence[Object]) -> bool:
        region, = objects
        scene: PackingScene = state.simulator_state
        box, _ = scene.regions[region.name]
        return abs(scene.dividers[box] - 0.5) < 1e-6

    @staticmethod
    def _free(state: State, objects: Sequence[Object]) -> bool:
        region, = objects
        return not state.simulator_state.contents[region.name]

    @staticmethod
    def _size_is(size: str) -> Callable[[State, Sequence[Object]], bool]:

        def _classifier(state: State, objects: Sequence[Object]) -> bool:
            item, = objects
            return state.simulator_state.item_sizes[item.name] == size

        return _classifier

    @staticmethod
    def _same_box(state: State, objects: Sequence[Object]) -> bool:
        region, other = objects
        scene: PackingScene = state.simulator_state
        return region != other and \
            scene.regions[region.name][0] == scene.regions[other.name][0]
