"""Cluttered Retrieval Sim: put the targets into the box, clearing the
obstacles that stand in front of them.

The simulated counterpart of the paper's Cluttered Retrieval Real domain:
"the robot retrieves targets into a box while reasoning about visual
obstruction. It removes pickable obstacles (e.g., bottles) or repositions
non-pickable ones (e.g., boxes)." A scene has one robot, one box (a
basket), targets (snack packs) and obstacles; every object is a vector of
features:

- robot: x, y (gripper position), gripper (1 while holding a target);
- target: x, y, in_box, on_table, held;
- obstacle: x, y, pickable (1 for a bottle, 0 for a bulky crate), cleared
  (1 once removed or pushed);
- box: x, y.

As in Table Clean Sim, the flags make the hidden predicates learnable from
few demonstrations; which target an obstacle blocks is left to the poses.

The table covers [0, 1] x [0, 1], the robot stands at y < 0 and the camera
looks from the far side (y > 1): an obstacle blocks a target on the table
if it stands in front of it as the camera sees it, between it and the
camera (up to corridor_length in y, within corridor_half_width in x). A
blocked target cannot be picked.

An action is [primitive, x, y, direction]: pick the target at (x, y), place
the held target at (x, y) (into the box), remove the pickable obstacle at
(x, y) (it is put aside, off the work area), or push the bulky obstacle at
(x, y) sideways by push_distance (direction > 0: toward larger x). An action
whose conditions do not hold changes nothing: the robot holds at most one
target, removing or pushing needs the hand empty, and a blocked target
stays where it is.

The goal is every target in the box. Train tasks have two targets, each
blocked by at most one obstacle; out-of-distribution test tasks have three
targets and more obstacles: a target may be blocked by two, and an obstacle
may block nothing.

Clear(target) is a derived predicate, ForAll ?obstacle. Not
Blocks(?obstacle, ?target): no operator adds it, so the ground-truth NSRTs
plan with an atoms closure (see
unipred/ground_truth_models/cluttered_retrieval/nsrts.py). Pickable and
Bulky are static.
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
PICK, PLACE, REMOVE, PUSH = range(4)


class ClutteredRetrievalSimEnv(BaseEnv):
    """Cluttered Retrieval Sim (see module doc)."""
    # The box, near the robot on its right.
    box_xy: ClassVar[Tuple[float, float]] = (0.85, 0.2)
    box_half_size: ClassVar[float] = 0.1
    # Targets rest in this region, at least target_gap apart in x (so that
    # a pushed obstacle blocks no other target).
    target_x: ClassVar[Tuple[float, float]] = (0.15, 0.75)
    target_y: ClassVar[Tuple[float, float]] = (0.2, 0.5)
    target_gap: ClassVar[float] = 0.25
    # An obstacle blocks a target if it is in front of it, toward the
    # camera: 0 < dy <= corridor_length and |dx| <= corridor_half_width.
    corridor_length: ClassVar[float] = 0.16
    corridor_half_width: ClassVar[float] = 0.06
    # Obstacles stand this far in front of the target they block (two
    # blocking one target: one near, one far), within max_offset in x.
    near_gap: ClassVar[Tuple[float, float]] = (0.05, 0.08)
    far_gap: ClassVar[Tuple[float, float]] = (0.12, 0.15)
    max_offset: ClassVar[float] = 0.03
    # Far enough to clear the target, not so far as to block the next.
    push_distance: ClassVar[float] = 0.15
    # Removed obstacles go to the far left corner, one after another.
    aside_x: ClassVar[float] = 0.05
    aside_y: ClassVar[Tuple[float, float]] = (0.9, 0.1)  # first, step
    # The gripper reaches objects within this distance of its target.
    reach: ClassVar[float] = 0.02
    robot_home: ClassVar[Tuple[float, float]] = (0.5, -0.1)

    robot_type = Type("robot", ["x", "y", "gripper"])
    target_type = Type("target", ["x", "y", "in_box", "on_table", "held"])
    obstacle_type = Type("obstacle", ["x", "y", "pickable", "cleared"])
    box_type = Type("box", ["x", "y"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        robot, target = self.robot_type, self.target_type
        obstacle, box = self.obstacle_type, self.box_type
        self._HandEmpty = Predicate("HandEmpty", [robot], self._hand_empty)
        self._Holding = Predicate("Holding", [robot, target], self._holding)
        self._OnTable = Predicate("OnTable", [target], self._on_table)
        self._InBox = Predicate("InBox", [target, box], self._in_box)
        self._Blocks = Predicate("Blocks", [obstacle, target], self._blocks)
        self._Clear = Predicate("Clear", [target], self._clear)
        self._Pickable = Predicate("Pickable", [obstacle], self._pickable)
        self._Bulky = Predicate("Bulky", [obstacle], self._bulky)

    @classmethod
    def get_name(cls) -> str:
        return "cluttered_retrieval_sim"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._Holding, self._OnTable, self._InBox,
            self._Blocks, self._Clear, self._Pickable, self._Bulky
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InBox}

    @property
    def derived_predicates(self) -> Set[Predicate]:
        return {self._Clear}

    @property
    def types(self) -> Set[Type]:
        return {
            self.robot_type, self.target_type, self.obstacle_type,
            self.box_type
        }

    @property
    def action_space(self) -> Box:
        return Box(np.array([0, 0, 0, -1], dtype=np.float32),
                   np.array([PUSH, 1, 1, 1], dtype=np.float32))

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        primitive = int(round(float(action.arr[0])))
        x, y, direction = (float(v) for v in action.arr[1:])
        robot = self._only(state, self.robot_type)
        held = self._held_target(state)
        next_state = state.copy()
        if primitive == PICK:
            target = self._at(state, self.target_type, x, y)
            if held is not None or target is None or \
                    not self._on_table(state, [target]) or \
                    not self._clear(state, [target]):
                return next_state
            self._set_place(next_state, target, x, y, "held")
            next_state.set(robot, "gripper", 1.0)
        elif primitive == PLACE:
            if held is None or not self._point_in_box(x, y):
                return next_state
            self._set_place(next_state, held, x, y, "in_box")
            next_state.set(robot, "gripper", 0.0)
        elif primitive in (REMOVE, PUSH):
            obstacle = self._at(state, self.obstacle_type, x, y)
            if held is not None or obstacle is None or (
                    state.get(obstacle, "pickable") > 0.5) != (primitive
                                                                == REMOVE):
                return next_state
            if primitive == REMOVE:
                x, y = self._aside_spot(state)
            else:
                x += self.push_distance * (1.0 if direction > 0 else -1.0)
                if not 0.0 <= x <= 1.0:
                    return state.copy()
            next_state.set(obstacle, "x", x)
            next_state.set(obstacle, "y", y)
            next_state.set(obstacle, "cleared", 1.0)
        else:
            return next_state
        next_state.set(robot, "x", x)
        next_state.set(robot, "y", y)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [
            self._make_task(CFG.cluttered_retrieval_sim_num_targets_train,
                            False, self._train_rng)
            for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if CFG.in_domain_test:
            return [
                self._make_task(
                    CFG.cluttered_retrieval_sim_num_targets_train, False,
                    self._test_rng) for _ in range(CFG.num_test_tasks)
            ]
        return [
            self._make_task(CFG.cluttered_retrieval_sim_num_targets_test,
                            True, self._test_rng)
            for _ in range(CFG.num_test_tasks)
        ]

    def _make_task(self, num_targets: int, cluttered: bool,
                   rng: np.random.Generator) -> EnvironmentTask:
        """Targets on the table with obstacles in front of them: in train
        tasks (not cluttered) each target blocked by one obstacle or none,
        at least one blocked; in test tasks two obstacles block one target
        (the farther a bottle), one obstacle or none each other target, and
        an obstacle may stand aside."""
        robot = Object("robot", self.robot_type)
        box = Object("box", self.box_type)
        targets = [
            Object(f"target{i}", self.target_type) for i in range(num_targets)
        ]
        blockers = [int(rng.random() < 0.75) for _ in targets]
        if not any(blockers):
            # At least one obstacle: Clear quantifies over the obstacles, and
            # as in IVNTR's classifiers a quantifier over none is false.
            blockers[int(rng.integers(num_targets))] = 1
        if cluttered:
            blockers = [int(rng.integers(0, 2)) for _ in targets]
            blockers[int(rng.integers(num_targets))] = 2
        xs = self._target_xs(num_targets, rng)
        data = {robot: [*self.robot_home, 0.0], box: list(self.box_xy)}
        obstacles: List[Tuple[float, float, float, float]] = []
        for target, x, count in zip(targets, xs, blockers):
            y = float(rng.uniform(*self.target_y))
            data[target] = [x, y, 0.0, 1.0, 0.0]
            gaps = [self.near_gap, self.far_gap][:count]
            for k, gap in enumerate(gaps):
                # The farther of two is a bottle.
                pickable = float(k == 1 or rng.random() < 0.5)
                obstacles.append(
                    (x + float(rng.uniform(-1, 1)) * self.max_offset,
                     y + float(rng.uniform(*gap)), pickable, 0.0))
        if cluttered and rng.random() < 0.5:
            # One that blocks nothing: behind the targets, by the box.
            obstacles.append((float(rng.uniform(0.85, 0.95)),
                              float(rng.uniform(0.6, 0.9)),
                              float(rng.random() < 0.5), 0.0))
        for i, features in enumerate(obstacles):
            data[Object(f"obstacle{i}", self.obstacle_type)] = list(features)
        state = State({
            obj: np.array(features, dtype=np.float32)
            for obj, features in data.items()
        })
        goal = {GroundAtom(self._InBox, [t, box]) for t in targets}
        return EnvironmentTask(state, goal)

    def _target_xs(self, num: int, rng: np.random.Generator) -> List[float]:
        """Target x positions in target_x, at least target_gap apart."""
        slack = self.target_x[1] - self.target_x[0] - (num -
                                                       1) * self.target_gap
        assert slack >= 0, "The targets do not fit."
        cuts = np.sort(rng.uniform(0, slack, size=num))
        xs = [self.target_x[0] + cut + i * self.target_gap
              for i, cut in enumerate(cuts)]
        return [float(x) for x in rng.permutation(xs)]

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="burlywood",
                                   alpha=0.3))
        h = self.box_half_size
        ax.add_patch(
            plt.Rectangle((self.box_xy[0] - h, self.box_xy[1] - h),
                          2 * h,
                          2 * h,
                          fill=False,
                          color="black"))
        for obj in sorted(state):
            if obj.type == self.target_type:
                size = 12 if state.get(obj, "held") > 0.5 else 8
                ax.plot(state.get(obj, "x"), state.get(obj, "y"), "gs",
                        markersize=size)
            elif obj.type == self.obstacle_type:
                marker = "bo" if state.get(obj, "pickable") > 0.5 else "ks"
                ax.plot(state.get(obj, "x"), state.get(obj, "y"), marker)
            else:
                continue
            ax.annotate(obj.name, (state.get(obj, "x"), state.get(obj, "y")))
        robot = self._only(state, self.robot_type)
        ax.plot(state.get(robot, "x"), state.get(robot, "y"), "k+",
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

    def _held_target(self, state: State) -> Optional[Object]:
        for target in state.get_objects(self.target_type):
            if state.get(target, "held") > 0.5:
                return target
        return None

    def _at(self, state: State, obj_type: Type, x: float,
            y: float) -> Optional[Object]:
        """The object of the type within reach of (x, y), if any."""
        best, best_dist = None, self.reach
        for obj in state.get_objects(obj_type):
            dist = float(
                np.hypot(state.get(obj, "x") - x,
                         state.get(obj, "y") - y))
            if dist <= best_dist:
                best, best_dist = obj, dist
        return best

    def _aside_spot(self, state: State) -> Tuple[float, float]:
        """The first free spot of the column of removed obstacles."""
        taken = {
            round(float(state.get(o, "y")), 3)
            for o in state.get_objects(self.obstacle_type)
            if abs(state.get(o, "x") - self.aside_x) < 1e-3
        }
        y = self.aside_y[0]
        while round(y, 3) in taken:
            y -= self.aside_y[1]
        return self.aside_x, y

    @staticmethod
    def _set_place(state: State, target: Object, x: float, y: float,
                   flag: str) -> None:
        state.set(target, "x", x)
        state.set(target, "y", y)
        for name in ("in_box", "on_table", "held"):
            state.set(target, name, float(name == flag))

    def _point_in_box(self, x: float, y: float) -> bool:
        return abs(x - self.box_xy[0]) <= self.box_half_size and \
            abs(y - self.box_xy[1]) <= self.box_half_size

    # Predicate classifiers.

    @staticmethod
    def _hand_empty(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "gripper") < 0.5

    @staticmethod
    def _holding(state: State, objects: Sequence[Object]) -> bool:
        _, target = objects
        return state.get(target, "held") > 0.5

    @staticmethod
    def _on_table(state: State, objects: Sequence[Object]) -> bool:
        target, = objects
        return state.get(target, "on_table") > 0.5

    @staticmethod
    def _in_box(state: State, objects: Sequence[Object]) -> bool:
        target, _ = objects
        return state.get(target, "in_box") > 0.5

    def _blocks(self, state: State, objects: Sequence[Object]) -> bool:
        obstacle, target = objects
        if not self._on_table(state, [target]):
            return False
        dx = state.get(obstacle, "x") - state.get(target, "x")
        dy = state.get(obstacle, "y") - state.get(target, "y")
        return 0 < dy <= self.corridor_length and \
            abs(dx) <= self.corridor_half_width

    def _clear(self, state: State, objects: Sequence[Object]) -> bool:
        target, = objects
        return not any(
            self._blocks(state, [obstacle, target])
            for obstacle in state.get_objects(self.obstacle_type))

    @staticmethod
    def _pickable(state: State, objects: Sequence[Object]) -> bool:
        obstacle, = objects
        return state.get(obstacle, "pickable") > 0.5

    @staticmethod
    def _bulky(state: State, objects: Sequence[Object]) -> bool:
        obstacle, = objects
        return state.get(obstacle, "pickable") < 0.5
