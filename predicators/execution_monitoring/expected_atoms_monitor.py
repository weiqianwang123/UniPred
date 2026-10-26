"""An execution monitor that leverages knowledge of the high-level plan to only
suggest replanning when the expected atoms check is not met.

The approach info is a list whose first entry is checked. It is either a
set of atoms that must hold in the current state (plans found without
simulation, one entry per step) or a pair (option, atoms), where the atoms
must hold once the running option terminates (plans found with simulation).
"""

import logging

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.structs import State


class ExpectedAtomsExecutionMonitor(BaseExecutionMonitor):
    """An execution monitor that only suggests replanning when we're doing
    bilevel planning and the expected atoms check fails."""

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms"

    def step(self, state: State) -> bool:
        # If the approach info is empty, don't replan.
        if not self._approach_info:  # pragma: no cover
            return False
        next_expected_atoms = self._approach_info[0]
        if isinstance(next_expected_atoms, tuple):
            # A plan found with simulation: the atoms only need to hold once
            # the running option terminates.
            option, next_expected_atoms = next_expected_atoms
            if not option.terminal(state):
                return False
        assert isinstance(next_expected_atoms, set)
        self._curr_plan_timestep += 1
        # If the expected atoms are a subset of the current atoms, then
        # we don't have to replan.
        unsat_atoms = {a for a in next_expected_atoms if not a.holds(state)}
        if not unsat_atoms:
            return False
        logging.info(
            "Expected atoms execution monitor triggered replanning "
            f"because of these atoms: {unsat_atoms}")  # pragma: no cover
        return True  # pragma: no cover
