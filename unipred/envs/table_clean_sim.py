"""Table Clean Sim: put the toys into the box, wipe the table and put the
wiper back.

This is the running example of the UniPred paper. A scene has one robot,
one table, one box, one wiper (a towel) and several toys. Every object is
a vector of features:

- robot: x, y (gripper position), gripper (1 while holding an object);
- toy: x, y, in_box, on_table, held;
- wiper: x, y, in_box, held;
- box: x, y, side (0 on the near side of the table, 1 on the far side);
- table: x, y (its center), is_clean.

An action is [primitive, x, y]: pick the resting object at (x, y), place
the held object at (x, y), push the box at (x, y) to the far side, pull it
to the near side, or wipe the table. An action whose conditions do not hold
changes nothing. The conditions, from the paper:

- the robot holds at most one object;
- wiping needs the wiper in hand, no toy on the table (wiping would sweep
  it away) and the box on the far side (it would be in the way);
- the wiper can be taken out of or put into the box only while the box is
  on the near side, within reach.

The goal is every toy in the box, the table clean and the wiper back in
the box. Train tasks start from the standard scene: two toys on the table,
the wiper in the box, the box on the near side and the hand empty. Out-of-
distribution test tasks have three toys and starts that the demonstrations
never show: the robot may already hold a toy or the wiper, toys may
already be in the box and the box may be on the far side.

NoToyOnTable(table) is a derived predicate, ForAll ?toy. Not
ToyOnTable(?toy, ?table): no operator adds it, so the ground-truth NSRTs
plan with an atoms closure (see
unipred/ground_truth_models/table_clean/nsrts.py).
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

# Primitives, the first entry of an action.
PICK, PLACE, PUSH_OUT, PULL_IN, WIPE = range(5)
# The feature that is 1 while a toy or the wiper is at a place.
_PLACE_FLAGS = {"table": "on_table", "box": "in_box", "hand": "held"}


class TableCleanSimEnv(BaseEnv):
    """Table Clean Sim (see module doc)."""
    # The table covers [0, 1] x [0, 1]; the robot stands at y < 0.
    table_center: ClassVar[float] = 0.5
    # The box is at x = box_x, on the near or the far side.
    box_x: ClassVar[float] = 0.85
    box_y: ClassVar[Tuple[float, float]] = (0.2, 0.8)  # near, far
    box_half_size: ClassVar[float] = 0.1
    # Toys on the table rest in this region, at least min_dist apart.
    free_x: ClassVar[Tuple[float, float]] = (0.05, 0.65)
    free_y: ClassVar[Tuple[float, float]] = (0.05, 0.95)
    min_dist: ClassVar[float] = 0.1
    # The gripper reaches objects within this distance of its target.
    reach: ClassVar[float] = 0.04
    # Where the robot's gripper rests at the start.
    robot_home: ClassVar[Tuple[float, float]] = (0.5, -0.1)

    robot_type = Type("robot", ["x", "y", "gripper"])
    toy_type = Type("toy", ["x", "y", "in_box", "on_table", "held"])
    wiper_type = Type("wiper", ["x", "y", "in_box", "held"])
    box_type = Type("box", ["x", "y", "side"])
    table_type = Type("table", ["x", "y", "is_clean"])

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
        self._WiperInBox = Predicate("WiperInBox", [wiper, box],
                                     self._in_box)
        self._BoxNear = Predicate("BoxNear", [box], self._box_near)
        self._BoxFar = Predicate("BoxFar", [box], self._box_far)
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
            self._ToyOnTable, self._ToyInBox, self._WiperInBox, self._BoxNear,
            self._BoxFar, self._TableClean, self._NoToyOnTable
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._ToyInBox, self._TableClean, self._WiperInBox}

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
        robot = self._only(state, self.robot_type)
        box = self._only(state, self.box_type)
        table = self._only(state, self.table_type)
        held = self._held_object(state)
        box_near = state.get(box, "side") < 0.5
        next_state = state.copy()
        if primitive == PICK:
            target = self._resting_object_at(state, x, y)
            if held is not None or target is None:
                return next_state
            if state.get(target, "in_box") > 0.5 and not box_near:
                return next_state
            self._set_place(next_state, target, x, y, "hand")
            next_state.set(robot, "gripper", 1.0)
        elif primitive == PLACE:
            if held is None:
                return next_state
            if self._point_in_box(state, x, y):
                if held.type == self.wiper_type and not box_near:
                    return next_state
                self._set_place(next_state, held, x, y, "box")
            elif held.type == self.toy_type and self._point_on_table(x, y):
                self._set_place(next_state, held, x, y, "table")
            else:
                return next_state
            next_state.set(robot, "gripper", 0.0)
        elif primitive in (PUSH_OUT, PULL_IN):
            if not self._near_point(state, box, x, y) or \
                    box_near != (primitive == PUSH_OUT):
                return next_state
            side = 1.0 if primitive == PUSH_OUT else 0.0
            new_y = self.box_y[int(side)]
            # Objects in the box move with it.
            for obj in state:
                if obj.type in (self.toy_type, self.wiper_type) and \
                        state.get(obj, "in_box") > 0.5:
                    next_state.set(
                        obj, "y",
                        state.get(obj, "y") - state.get(box, "y") + new_y)
            next_state.set(box, "y", new_y)
            next_state.set(box, "side", side)
        elif primitive == WIPE:
            if held is None or held.type != self.wiper_type or box_near or \
                    not self._no_toy_on_table(state, [table]):
                return next_state
            next_state.set(table, "is_clean", 1.0)
        else:
            return next_state
        next_state.set(robot, "x", x)
        next_state.set(robot, "y", y)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [
            self._standard_task(CFG.table_clean_sim_num_toys_train,
                                self._train_rng)
            for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            return [
                self._standard_task(CFG.table_clean_sim_num_toys_train,
                                    self._test_rng)
                for _ in range(CFG.num_test_tasks)
            ]
        return [
            self._varied_task(CFG.table_clean_sim_num_toys_test,
                              self._test_rng)
            for _ in range(CFG.num_test_tasks)
        ]

    def _standard_task(self, num_toys: int,
                       rng: np.random.Generator) -> EnvironmentTask:
        """Toys on the table, the wiper in the box near the robot, the hand
        empty."""
        return self._make_task(["table"] * num_toys, "box", 0.0, rng)

    def _varied_task(self, num_toys: int,
                     rng: np.random.Generator) -> EnvironmentTask:
        """Toys on the table or already in the box, the box on either side,
        and possibly a toy or the wiper in hand."""
        toy_places = [
            "box" if rng.random() < 0.3 else "table" for _ in range(num_toys)
        ]
        wiper_place = "box"
        in_hand = rng.choice(["none", "toy", "wiper"])
        if in_hand == "toy":
            toy_places[int(rng.integers(num_toys))] = "hand"
        elif in_hand == "wiper":
            wiper_place = "hand"
        side = float(rng.random() < 0.5)
        return self._make_task(toy_places, wiper_place, side, rng)

    def _make_task(self, toy_places: Sequence[str], wiper_place: str,
                   side: float,
                   rng: np.random.Generator) -> EnvironmentTask:
        robot = Object("robot", self.robot_type)
        table = Object("table", self.table_type)
        box = Object("box", self.box_type)
        wiper = Object("wiper", self.wiper_type)
        toys = [
            Object(f"toy{i}", self.toy_type) for i in range(len(toy_places))
        ]
        box_y = self.box_y[int(side)]
        table_spots = iter(
            self._free_spots(sum(p == "table" for p in toy_places), rng))
        hand_xy = self.robot_home
        in_hand = "hand" in list(toy_places) + [wiper_place]
        if in_hand:
            hand_xy = (self.table_center, 0.0)

        def _features(obj: Object, place: str) -> List[float]:
            """The features of a toy or the wiper on the table, in the box
            or in the hand."""
            if place == "table":
                xy = next(table_spots)
            elif place == "box":
                offset = rng.uniform(-0.6, 0.6, size=2) * self.box_half_size
                xy = (self.box_x + offset[0], box_y + offset[1])
            else:
                xy = hand_xy
            flag = _PLACE_FLAGS[place]
            return [*xy] + [
                float(name == flag) for name in obj.type.feature_names[2:]
            ]

        data = {toy: _features(toy, place) for toy, place in zip(toys, toy_places)}
        data[wiper] = _features(wiper, wiper_place)
        data[box] = [self.box_x, box_y, side]
        data[table] = [self.table_center, self.table_center, 0.0]
        data[robot] = [*hand_xy, float(in_hand)]
        state = State({
            obj: np.array(features, dtype=np.float32)
            for obj, features in data.items()
        })
        goal = {GroundAtom(self._ToyInBox, [toy, box]) for toy in toys}
        goal.add(GroundAtom(self._TableClean, [table]))
        goal.add(GroundAtom(self._WiperInBox, [wiper, box]))
        return EnvironmentTask(state, goal)

    def _free_spots(self, num: int,
                    rng: np.random.Generator) -> List[Tuple[float, float]]:
        """Rejection-sample positions on the table that are far apart."""
        spots: List[Tuple[float, float]] = []
        while len(spots) < num:
            spot = (float(rng.uniform(*self.free_x)),
                    float(rng.uniform(*self.free_y)))
            if all(
                    np.hypot(spot[0] - s[0], spot[1] - s[1]) >= self.min_dist
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
        clean = state.get(table, "is_clean") > 0.5
        ax.add_patch(
            plt.Rectangle((0, 0),
                          1,
                          1,
                          color="burlywood" if clean else "saddlebrown",
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
                size = 12 if state.get(obj, "held") > 0.5 else 8
                ax.plot(state.get(obj, "x"),
                        state.get(obj, "y"),
                        marker,
                        markersize=size)
                ax.annotate(obj.name,
                            (state.get(obj, "x"), state.get(obj, "y")))
        robot = self._only(state, self.robot_type)
        ax.plot(state.get(robot, "x"),
                state.get(robot, "y"),
                "k+",
                markersize=15)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 1.1)
        ax.set_aspect("equal")
        title = f"goal: {sorted(task.goal)}"
        if action is not None:
            title += f"\naction: {action.arr}"
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
                    state.get(obj, "held") > 0.5:
                return obj
        return None

    def _resting_object_at(self, state: State, x: float,
                           y: float) -> Optional[Object]:
        """The toy or wiper within reach of (x, y) that is not held."""
        best, best_dist = None, self.reach
        for obj in sorted(state):
            if obj.type not in (self.toy_type, self.wiper_type) or \
                    state.get(obj, "held") > 0.5:
                continue
            dist = float(
                np.hypot(state.get(obj, "x") - x,
                         state.get(obj, "y") - y))
            if dist <= best_dist:
                best, best_dist = obj, dist
        return best

    @staticmethod
    def _set_place(state: State, obj: Object, x: float, y: float,
                   place: str) -> None:
        """Move a toy or the wiper to (x, y) on the "table", in the "box" or
        in the "hand"."""
        state.set(obj, "x", x)
        state.set(obj, "y", y)
        for name in obj.type.feature_names[2:]:
            state.set(obj, name, float(name == _PLACE_FLAGS[place]))

    def _point_in_box(self, state: State, x: float, y: float) -> bool:
        box = self._only(state, self.box_type)
        return abs(x - state.get(box, "x")) <= self.box_half_size and \
            abs(y - state.get(box, "y")) <= self.box_half_size

    def _point_on_table(self, x: float, y: float) -> bool:
        return self.free_x[0] <= x <= self.free_x[1] and \
            self.free_y[0] <= y <= self.free_y[1]

    def _near_point(self, state: State, obj: Object, x: float,
                    y: float) -> bool:
        return float(
            np.hypot(state.get(obj, "x") - x,
                     state.get(obj, "y") - y)) <= self.reach

    # Predicate classifiers.

    @staticmethod
    def _hand_empty(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "gripper") < 0.5

    @staticmethod
    def _holding(state: State, objects: Sequence[Object]) -> bool:
        _, obj = objects
        return state.get(obj, "held") > 0.5

    @staticmethod
    def _on_table(state: State, objects: Sequence[Object]) -> bool:
        toy, _ = objects
        return state.get(toy, "on_table") > 0.5

    @staticmethod
    def _in_box(state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        return state.get(obj, "in_box") > 0.5

    @staticmethod
    def _box_near(state: State, objects: Sequence[Object]) -> bool:
        box, = objects
        return state.get(box, "side") < 0.5

    @staticmethod
    def _box_far(state: State, objects: Sequence[Object]) -> bool:
        box, = objects
        return state.get(box, "side") > 0.5

    @staticmethod
    def _clean(state: State, objects: Sequence[Object]) -> bool:
        table, = objects
        return state.get(table, "is_clean") > 0.5

    def _no_toy_on_table(self, state: State,
                         objects: Sequence[Object]) -> bool:
        table, = objects
        return not any(
            self._on_table(state, [toy, table])
            for toy in state.get_objects(self.toy_type))
