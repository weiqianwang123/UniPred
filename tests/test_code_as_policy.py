"""Tests for the Code as Policy approach."""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from unipred.approaches.code_as_policy import CodeAsPolicyApproach, \
    PolicyError, compile_policy, extract_code

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FIRST_BLOCK_POLICY = '''
def policy(state, goal):
    blocks = sorted(n for n, o in state.items() if o["type"] == "block")
    return "PickFromTable", ["robby", blocks[0]], []
'''


def test_extract_and_compile():
    """Code is taken from the first fenced block; errors name the line."""
    assert extract_code("Here:\n```python\nx = 1\n```\nDone") == "x = 1\n"
    assert extract_code("x = 1") == "x = 1\n"
    assert compile_policy("def policy(state, goal):\n    return 1\n")({},
                                                                       []) == 1
    with pytest.raises(PolicyError, match="does not define"):
        compile_policy("x = 1\n")
    with pytest.raises(PolicyError, match="line 2"):
        compile_policy("x = 1\ny = 1 / 0\n")


def _approach():
    utils.reset_config({
        "env": "blocks_onclear",
        "num_train_tasks": 1,
        "neupi_pred_config": os.path.join(_REPO,
                                          "configs/blocks/predicates.yaml"),
    })
    env = get_or_create_env("blocks_onclear")
    tasks = [t.task for t in env.get_train_tasks()]
    approach = CodeAsPolicyApproach(env.predicates,
                                    get_gt_options(env.get_name()),
                                    env.types, env.action_space, tasks)
    return approach, tasks[0]


def test_next_option_checks_the_returned_skill():
    """The skill a policy returns is checked and grounded; invalid ones
    raise PolicyError with a message for the LLM."""
    approach, task = _approach()
    # pylint: disable=protected-access
    option = approach._next_option(compile_policy(_FIRST_BLOCK_POLICY),
                                   task.init, task.goal)
    assert option.name == "PickFromTable"
    assert [o.name for o in option.objects] == ["robby", "block0"]
    for returned, message in [
        ('"Fly", [], []', "no skill"),
        ('"PickFromTable", ["robby", "ghost"], []', "not all object names"),
        ('"PickFromTable", ["block0", "robby"], []', "types"),
        ('"PutOnTable", ["robby", "block0"], [0.5]', "2 finite"),
        ('"PickFromTable"', "instead of"),
    ]:
        policy = compile_policy(f"def policy(state, goal):\n"
                                f"    return {returned}\n")
        with pytest.raises(PolicyError, match=message):
            approach._next_option(policy, task.init, task.goal)
    # Parameters are clipped to the skill's parameter space.
    policy = compile_policy('def policy(state, goal):\n'
                            '    return "PutOnTable", ["robby", "block0"], '
                            '[2.0, -1.0]\n')
    put = approach._next_option(policy, task.init, task.goal)
    assert np.allclose(put.params, [1.0, 0.0])


def test_repair_after_an_error(tmp_path):
    """A policy that raises an error on a training task is shown to the LLM
    with the error; every version is saved and the repaired one is used."""
    responses = tmp_path / "responses.json"
    responses.write_text(
        json.dumps([{
            "task": "code_as_policy",
            "contains": "Your previous code",
            "response": f"```python\n{_FIRST_BLOCK_POLICY}```"
        }, {
            "task": "code_as_policy",
            "response": "def policy(state, goal):\n    return state['nope']\n"
        }]))
    out = str(tmp_path)
    command = [
        sys.executable, "-m", "unipred.main", "--config",
        "configs/blocks/code_as_policy.yaml", "--config",
        "configs/llm/mock.yaml", "--llm_mock_responses",
        str(responses), "--seed", "0", "--num_train_tasks", "2",
        "--num_test_tasks", "1", "--approach_dir", out, "--data_dir", out,
        "--results_dir", out, "--eval_trajectories_dir", out,
        "--llm_cache_dir", "", "--log_file",
        os.path.join(out, "log.txt")
    ]
    result = subprocess.run(command, cwd=_REPO, capture_output=True,
                            text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    log = (tmp_path / "log.txt").read_text()
    assert "Policy version 0 failed" in log
    assert "Policy version 1 solves" in log
    assert "Tasks solved:" in log
    versions = sorted(glob.glob(os.path.join(out, "*.policy_*.py")))
    assert len(versions) == 2
    final = glob.glob(os.path.join(out, "*.saved.policy.py"))[0]
    with open(final, encoding="utf-8") as f:
        assert "PickFromTable" in f.read()
    with open(glob.glob(os.path.join(out, "*.llm_log.jsonl"))[0],
              encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    assert len(requests) == 2
    assert "KeyError: 'nope' (line 2)" in requests[1]["user"]
