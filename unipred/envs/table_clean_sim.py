"""Table Clean Sim: put every toy into the box and wipe the table.

This is the running example of the UniPred paper. A scene has one robot,
one table, one box, one wiper (a towel) and several toys. The state holds
2D poses and a few attributes per object:

- robot: x, y (gripper position), gripper (1 while holding an object);
- toy, wiper: x, y, z (1 while lifted by the robot, 0 when resting);
- box: x, y; the box is either on the near side of the table (close to
  the robot) or on the far side;
- table: x, y (its center), dirt (1 while dirty, 0 once wiped).

An action is [primitive, x, y]. The primitives pick the resting object
nearest to (x, y), place the held object at (x, y), push the box at (x, y)
to the far side, pull it back to the near side, or wipe the table. An
action whose conditions do not hold changes nothing. The conditions, from
the paper:

- the robot holds at most one object, and pushes or pulls the box only
  with an empty hand;
- wiping needs the wiper in hand, no toy on the table (wiping would sweep
  it away) and the box on the far side (it would be in the way);
- the wiper can be taken out of or put into the box only while the box is
  on the near side.

Train tasks start from a standard scene: two toys on the table, the wiper
in the box, the box on the near side and the hand empty. Out-of-
distribution test tasks have three toys and starts that the demonstrations
never show: the robot may already hold the wiper or a toy, toys may already
be in the box, the box may start on the far side and the wiper on the
table. The goal is always every toy in the box and the table clean.

NoToyOnTable(table) is a derived predicate: no operator adds it directly,
so the ground-truth NSRTs plan with an atoms closure (see
unipred/ground_truth_models/table_clean/nsrts.py).
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

# Primitives, the first entry of an action.
PICK, PLACE, PUSH_OUT, PULL_IN, WIPE = range(5)


class TableCleanSimEnv(BaseEnv):
    """Table Clean Sim (see module doc)."""
    # The table covers [0, 1] x [0, 1]; the robot stands at y < 0.
    table_center: ClassVar[float] = 0.5
    # The box column is at x = box_x, near or far from the robot.
    box_x: ClassVar[float] = 0.85
    box_near_y: ClassVar[float] = 0.2
    box_far_y: ClassVar[float] = 0.8
    box_half_size: ClassVar[float] = 0.1
    # Objects on the table rest in this region, at least min_dist apart.
    free_x: ClassVar[tuple] = (0.05, 0.65)
    free_y: ClassVar[tuple] = (0.05, 0.95)
    min_dist: ClassVar[float] = 0.1
    # The gripper reaches objects within this distance of its target.
    reach: ClassVar[float] = 0.04

    robot_type = Type("robot", ["x", "y", "gripper"])
    toy_type = Type("toy", ["x", "y", "z"])
    wiper_type = Type("wiper", ["x", "y", "z"])
    box_type = Type("box", ["x", "y"])
    table_type = Type("table", ["x", "y", "dirt"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        robot, toy, wiper = self.robot_type, self.toy_type, self.wiper_type
        box, table = self.box_type, self.table_type
        self._HandEmpty = Predicate("HandEmpty", [robot], self._hand_empty)
        self._HoldingToy = Predicate("HoldingToy", [robot, toy],
                                     self._holding)
        self._HoldingWiper = Predicate("HoldingWiper", [robot, wiper],
                                       self._holding)
        self._ToyOnTable = Predicate("ToyOnTable", [toy, table],
                                     self._on_table)
        self._ToyInBox = Predicate("ToyInBox", [toy, box], self._in_box)
        self._WiperOnTable = Predicate("WiperOnTable", [wiper, table],
                                       self._on_table)
        self._WiperInBox = Predicate("WiperInBox", [wiper, box],
                                     self._in_box)
        self._BoxNear = Predicate("BoxNear", [box, table], self._box_near)
        self._BoxFar = Predicate("BoxFar", [box, table], self._box_far)
        self._TableClean = Predicate("TableClean", [table], self._clean)
        self._NoToyOnTable = Predicate("NoToyOnTable", [table],
                                       self._no_toy_on_table)

    @classmethod
    def get_name(cls) -> str:
        return "table_clean_sim"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._HoldingToy, self._HoldingWiper,
            self._ToyOnTable, self._ToyInBox, self._WiperOnTable,
            self._WiperInBox, self._BoxNear, self._BoxFar, self._TableClean,
            self._NoToyOnTable
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._ToyInBox, self._TableClean}

    @property
    def derived_predicates(self) -> Set[Predicate]:
        return {self._NoToyOnTable}

    @property
    def types(self) -> Set[Type]:
        return {
            self.robot_type, self.toy_type, self.wiper_type, self.box_type,
            self.table_type
        }

    @property
    def action_space(self) -> Box:
        return Box(np.array([0, 0, 0], dtype=np.float32),
                   np.array([WIPE, 1, 1], dtype=np.float32))

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        primitive = int(round(float(action.arr[0])))
        x, y = float(action.arr[1]), float(action.arr[2])
        next_state = state.copy()
        robot = self._only(state, self.robot_type)
        box = self._only(state, self.box_type)
        table = self._only(state, self.table_type)
        held = self._held_object(state)
        if primitive == PICK:
            target = self._resting_object_at(state, x, y)
            if held is not None or target is None:
                return next_state
            if self._inside_box(state, target) and not self._is_near(
                    state, box):
                return next_state  # the far side is out of reach
            next_state.set(target, "x", x)
            next_state.set(target, "y", y)
            next_state.set(target, "z", 1.0)
            next_state.set(robot, "gripper", 1.0)
        elif primitive == PLACE:
            if held is None:
                return next_state
            if self._point_in_box(state, x, y) and \
                    held.type == self.wiper_type and \
                    not self._is_near(state, box):
                return next_state
            next_state.set(held, "x", x)
            next_state.set(held, "y", y)
            next_state.set(held, "z", 0.0)
            next_state.set(robot, "gripper", 0.0)
        elif primitive in (PUSH_OUT, PULL_IN):
            if held is not None or not self._near_point(state, box, x, y):
                return next_state
            if self._is_near(state, box) != (primitive == PUSH_OUT):
                return next_state
            new_y = self.box_far_y if primitive == PUSH_OUT \
                else self.box_near_y
            # Objects in the box move with it.
            for obj in state:
                if obj.type in (self.toy_type, self.wiper_type) and \
                        self._inside_box(state, obj):
                    next_state.set(obj, "y",
                                   state.get(obj, "y") - state.get(box, "y") +
                                   new_y)
            next_state.set(box, "y", new_y)
        elif primitive == WIPE:
            if held is None or held.type != self.wiper_type or \
                    self._is_near(state, box) or \
                    not self._no_toy_on_table(state, [table]):
                return next_state
            next_state.set(table, "dirt", 0.0)
        else:
            return next_state
        next_state.set(robot, "x", x)
        next_state.set(robot, "y", y)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [
            self._standard_task(CFG.table_clean_num_toys_train,
                                self._train_rng)
            for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            return [
                self._standard_task(CFG.table_clean_num_toys_train,
                                    self._test_rng)
                for _ in range(CFG.num_test_tasks)
            ]
        return [
            self._varied_task(CFG.table_clean_num_toys_test, self._test_rng)
            for _ in range(CFG.num_test_tasks)
        ]

    def _standard_task(self, num_toys: int,
                       rng: np.random.Generator) -> EnvironmentTask:
        """Toys on the table, the wiper in the box near the robot."""
        toy_places = ["table"] * num_toys
        return self._make_task(toy_places, "box", box_near=True, rng=rng)

    def _varied_task(self, num_toys: int,
                     rng: np.random.Generator) -> EnvironmentTask:
        """Toys on the table or in the box, the wiper on the table or in the
        box, the box near or far, and possibly an object in hand; at least
        one toy starts out of the box."""
        toy_places = [
            "box" if rng.random() < 0.3 else "table" for _ in range(num_toys)
        ]
        toy_places[int(rng.integers(num_toys))] = "table"
        wiper_place = "box" if rng.random() < 0.5 else "table"
        in_hand = rng.choice(["none", "wiper", "toy"])
        if in_hand == "wiper":
            wiper_place = "hand"
        elif in_hand == "toy":
            toy_places[toy_places.index("table")] = "hand"
        return self._make_task(toy_places,
                               wiper_place,
                               box_near=bool(rng.random() < 0.5),
                               rng=rng)

    def _make_task(self, toy_places: Sequence[str], wiper_place: str,
                   box_near: bool,
                   rng: np.random.Generator) -> EnvironmentTask:
        robot = Object("robot", self.robot_type)
        table = Object("table", self.table_type)
        box = Object("box", self.box_type)
        wiper = Object("wiper", self.wiper_type)
        toys = [Object(f"toy{i}", self.toy_type) for i in range(len(toy_places))]
        box_y = self.box_near_y if box_near else self.box_far_y
        resting = sum(place == "table" for place in toy_places) + int(
            wiper_place == "table")
        table_spots = iter(self._free_spots(resting, rng))
        robot_xy = (0.5, -0.1)
        gripper = 0.0

        def _pose(place: str) -> List[float]:
            nonlocal robot_xy, gripper
            if place == "table":
                return [*next(table_spots), 0.0]
            if place == "box":
                offset = rng.uniform(-0.6, 0.6, size=2) * self.box_half_size
                return [self.box_x + offset[0], box_y + offset[1], 0.0]
            assert place == "hand"
            robot_xy = (0.5, 0.0)
            gripper = 1.0
            return [*robot_xy, 1.0]

        data: Dict[Object, List[float]] = {
            toy: _pose(place)
            for toy, place in zip(toys, toy_places)
        }
        data[wiper] = _pose(wiper_place)
        data[box] = [self.box_x, box_y]
        data[table] = [self.table_center, self.table_center, 1.0]
        data[robot] = [*robot_xy, gripper]
        state = State({obj: np.array(v, dtype=np.float32)
                       for obj, v in data.items()})
        goal = {GroundAtom(self._ToyInBox, [toy, box]) for toy in toys}
        goal.add(GroundAtom(self._TableClean, [table]))
        return EnvironmentTask(state, goal)

    def _free_spots(self, num: int,
                    rng: np.random.Generator) -> List[tuple]:
        """Rejection-sample positions on the table that are far apart."""
        spots: List[tuple] = []
        while len(spots) < num:
            spot = (rng.uniform(*self.free_x), rng.uniform(*self.free_y))
            if all(np.hypot(spot[0] - s[0], spot[1] - s[1]) >= self.min_dist
                   for s in spots):
                spots.append(spot)
        return spots

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        table = self._only(state, self.table_type)
        box = self._only(state, self.box_type)
        ax.add_patch(
            plt.Rectangle((0, 0),
                          1,
                          1,
                          color="saddlebrown" if state.get(table, "dirt") >
                          0.5 else "burlywood",
                          alpha=0.4))
        h = self.box_half_size
        ax.add_patch(
            plt.Rectangle((state.get(box, "x") - h, state.get(box, "y") - h),
                          2 * h,
                          2 * h,
                          fill=False,
                          color="black"))
        for obj in sorted(state):
            if obj.type in (self.toy_type, self.wiper_type):
                marker = "o" if obj.type == self.toy_type else "s"
                ax.plot(state.get(obj, "x"),
                        state.get(obj, "y"),
                        marker,
                        markersize=12 if state.get(obj, "z") > 0.5 else 8)
                ax.annotate(obj.name, (state.get(obj, "x"), state.get(obj, "y")))
        robot = self._only(state, self.robot_type)
        ax.plot(state.get(robot, "x"), state.get(robot, "y"), "k+",
                markersize=15)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 1.1)
        ax.set_aspect("equal")
        title = f"goal: {sorted(task.goal)}"
        if caption is not None:
            title += f"\n{caption}"
        ax.set_title(title, fontsize=6)
        plt.tight_layout()
        return fig

    @staticmethod
    def _only(state: State, obj_type: Type) -> Object:
        objects = state.get_objects(obj_type)
        assert len(objects) == 1, f"Expected one {obj_type.name}."
        return objects[0]

    def _held_object(self, state: State) -> Optional[Object]:
        for obj in state:
            if obj.type in (self.toy_type, self.wiper_type) and \
                    state.get(obj, "z") > 0.5:
                return obj
        return None

    def _resting_object_at(self, state: State, x: float,
                           y: float) -> Optional[Object]:
        best, best_dist = None, self.reach
        for obj in sorted(state):
            if obj.type not in (self.toy_type, self.wiper_type) or \
                    state.get(obj, "z") > 0.5:
                continue
            dist = float(np.hypot(state.get(obj, "x") - x,
                                  state.get(obj, "y") - y))
            if dist <= best_dist:
                best, best_dist = obj, dist
        return best

    def _point_in_box(self, state: State, x: float, y: float) -> bool:
        box = self._only(state, self.box_type)
        return abs(x - state.get(box, "x")) <= self.box_half_size and \
            abs(y - state.get(box, "y")) <= self.box_half_size

    def _inside_box(self, state: State, obj: Object) -> bool:
        return state.get(obj, "z") < 0.5 and self._point_in_box(
            state, state.get(obj, "x"), state.get(obj, "y"))

    def _near_point(self, state: State, obj: Object, x: float,
                    y: float) -> bool:
        return float(np.hypot(state.get(obj, "x") - x,
                              state.get(obj, "y") - y)) <= self.reach

    def _is_near(self, state: State, box: Object) -> bool:
        return state.get(box, "y") < self.table_center

    # Predicate classifiers.

    @staticmethod
    def _hand_empty(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "gripper") < 0.5

    @staticmethod
    def _holding(state: State, objects: Sequence[Object]) -> bool:
        robot, obj = objects
        return state.get(robot, "gripper") > 0.5 and state.get(obj, "z") > 0.5

    def _on_table(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        return state.get(obj, "z") < 0.5 and not self._inside_box(state, obj)

    def _in_box(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        return self._inside_box(state, obj)

    def _box_near(self, state: State, objects: Sequence[Object]) -> bool:
        box, _ = objects
        return self._is_near(state, box)

    def _box_far(self, state: State, objects: Sequence[Object]) -> bool:
        box, _ = objects
        return not self._is_near(state, box)

    @staticmethod
    def _clean(state: State, objects: Sequence[Object]) -> bool:
        table, = objects
        return state.get(table, "dirt") < 0.5

    def _no_toy_on_table(self, state: State,
                         objects: Sequence[Object]) -> bool:
        table, = objects
        return not any(
            self._on_table(state, [toy, table])
            for toy in state.get_objects(self.toy_type))
