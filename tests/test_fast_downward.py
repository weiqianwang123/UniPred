"""Tests for planning with Fast Downward (needs FD_EXEC_PATH)."""

import os

import pytest

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs.satellites import SatellitesEnv
from predicators.ground_truth_models import get_gt_options


@pytest.mark.skipif("FD_EXEC_PATH" not in os.environ,
                    reason="Fast Downward is not installed")
@pytest.mark.parametrize("planner", ["fdsat", "fdopt"])
def test_planning_leaves_the_working_directory_alone(tmp_path, monkeypatch,
                                                     planner):
    """Fast Downward writes nothing to the working directory, so processes
    planning in the same directory do not delete each other's plans; its
    temporary files are removed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    import tempfile  # pylint: disable=import-outside-toplevel
    monkeypatch.setattr(tempfile, "tempdir", None)
    utils.reset_config({
        "env": "satellites",
        "num_train_tasks": 1,
        "sesame_task_planner": planner
    })
    env = SatellitesEnv()
    task = env.get_train_tasks()[0].task
    approach = OracleApproach(env.predicates, get_gt_options(env.get_name()),
                              env.types, env.action_space, [task])
    approach.solve(task, timeout=30)
    assert sorted(os.listdir(tmp_path)) == ["tmp"]
    assert not os.listdir(tmp_path / "tmp")
