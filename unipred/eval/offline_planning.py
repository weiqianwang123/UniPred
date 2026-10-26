"""Offline evaluation by task planning (the protocol of the paper's Table
III), for environments without a simulator.

For each test scene, the approach plans from the scene's initial state; the
skeleton it returns (its options and their objects) is then checked
symbolically: starting from the scene's annotated initial atoms, every step
must match a ground-truth operator whose preconditions hold, its effects
are applied (and derived atoms recomputed by the ground-truth closure), and
the goal must hold at the end. The scene counts as solved if the skeleton
is valid.

Approaches that plan with operators do so without a simulator
(--bilevel_plan_without_sim True) and expose their skeleton with
get_last_nsrt_plan(); the VLM planners (unipred/approaches/vila.py) expose
it with get_last_skeleton().
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence, Set, Tuple

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.envs import BaseEnv
from predicators.ground_truth_models import get_gt_atoms_closure, \
    get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import NSRT, GroundAtom, Metrics, Object, \
    ParameterizedOption, Task

Skeleton = List[Tuple[ParameterizedOption, Sequence[Object]]]


def last_skeleton(approach: Any) -> Skeleton:
    """The skeleton found by the approach's last call of solve()."""
    if isinstance(approach, BilevelPlanningApproach):
        return [(n.option, n.option_objs)
                for n in approach.get_last_nsrt_plan()]
    return approach.get_last_skeleton()


def check_skeleton(skeleton: Skeleton, init_atoms: Set[GroundAtom],
                   goal: Set[GroundAtom], objects: Sequence[Object],
                   nsrts: Set[NSRT],
                   closure: Any = None) -> Optional[str]:
    """Why the skeleton does not reach the goal from the initial atoms
    under the operators, or None if it does."""
    atoms = set(init_atoms)
    if closure is not None:
        atoms = closure(atoms, set(objects))
    for step, (option, option_objs) in enumerate(skeleton):
        candidates = [
            ground for nsrt in sorted(nsrts)
            if nsrt.option.name == option.name
            for ground in utils.all_ground_nsrts(nsrt, objects)
            if list(ground.option_objs) == list(option_objs)
        ]
        applicable = [g for g in candidates if g.preconditions <= atoms]
        name = f"{option.name}({', '.join(o.name for o in option_objs)})"
        if not applicable:
            if not candidates:
                return f"step {step + 1}, {name}: no such operator"
            missing = min((g.preconditions - atoms for g in candidates),
                          key=len)
            return (f"step {step + 1}, {name}: preconditions "
                    f"{sorted(missing)} do not hold")
        atoms = utils.apply_operator(applicable[0], atoms)
        if closure is not None:
            atoms = closure(atoms, set(objects))
    if not goal <= atoms:
        return f"the goal atoms {sorted(goal - atoms)} do not hold at the end"
    return None


def evaluate(env: BaseEnv, approach: Any, tasks: Sequence[Task]) -> Metrics:
    """Plan each task with the approach and check the skeleton with the
    environment's ground-truth operators (see module doc)."""
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    closure = get_gt_atoms_closure(env.get_name(), env.predicates)
    num_solved = 0
    total_time = 0.0
    for i, task in enumerate(tasks):
        start = time.perf_counter()
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachFailure, ApproachTimeout) as err:
            logging.info(f"Task {i + 1} / {len(tasks)}: no plan: {err}")
            continue
        plan_time = time.perf_counter() - start
        skeleton = last_skeleton(approach)
        init_atoms = utils.abstract(task.init, env.predicates)
        problem = check_skeleton(skeleton, init_atoms, task.goal,
                                 list(task.init), nsrts, closure)
        if problem is None:
            num_solved += 1
            total_time += plan_time
            logging.info(f"Task {i + 1} / {len(tasks)}: valid plan of "
                         f"length {len(skeleton)}")
        else:
            logging.info(f"Task {i + 1} / {len(tasks)}: invalid plan: "
                         f"{problem}")
    return {
        "num_solved": num_solved,
        "num_total": len(tasks),
        "avg_suc_time": total_time / num_solved if num_solved else float("inf"),
    }
