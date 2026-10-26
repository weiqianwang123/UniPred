"""End-to-end smoke tests: UniPred on a few tiny tasks of each domain with
the mock LLM (data, PDDL completion, LLM loop, selection, operator learning
and evaluation)."""

import glob
import os
import subprocess
import sys

import dill as pkl
import pytest
import torch
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TINY = dict(epochs=2, val_freq=1, num_iter=1, batch_vect_num=1,
             batch_size=32, guidance_thresh=1.0, loss_thresh=0.9)


def _run(tmp_path, domain, num_train_tasks, extra_flags):
    with open(os.path.join(_REPO, f"configs/{domain}/predicates.yaml"),
              encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["defaults"].update(_TINY)
    for entry in config["predicate_types"]:
        for key in _TINY:
            entry.pop(key, None)
    predicates = tmp_path / "predicates.yaml"
    predicates.write_text(yaml.safe_dump(config))
    out = str(tmp_path)
    command = [
        sys.executable, "-m", "unipred.main", "--config",
        f"configs/{domain}/unipred.yaml", "--config", "configs/llm/mock.yaml",
        "--neupi_pred_config",
        str(predicates), "--seed", "0", "--num_train_tasks",
        str(num_train_tasks), "--num_test_tasks", "1", "--device", "cpu",
        "--approach_dir", out, "--neupi_save_path", out, "--data_dir", out,
        "--results_dir", out, "--eval_trajectories_dir", out,
        "--llm_cache_dir", "", "--log_file",
        os.path.join(out, "log.txt"), "--sesame_task_planner", "astar"
    ] + extra_flags
    result = subprocess.run(command, cwd=_REPO, capture_output=True,
                            text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    log = (tmp_path / "log.txt").read_text()
    assert "Proposing PDDL seed" in log
    assert "Tasks solved:" in log
    assert any(name.endswith(".llm_history.json") for name in os.listdir(out))


@pytest.mark.parametrize("domain, num_train_tasks", [
    ("satellites", 10),
    ("blocks", 10),
    ("table_clean_sim", 5),
])
def test_unipred_mock(tmp_path, domain, num_train_tasks):
    """UniPred runs end to end on each domain."""
    _run(tmp_path, domain, num_train_tasks, [])


def test_unipred_mock_derived_selection(tmp_path):
    """The run also completes with derived-aware selection (whose logic is
    tested in test_derived_selection.py)."""
    _run(tmp_path, "satellites", 10, ["--unipred_derived_selection", "True"])


def _nested_lists(value):
    if torch.is_tensor(value):
        return value.tolist()
    return [_nested_lists(v) for v in value]


def _learned(out):
    """What a run learned: the kept effect vectors of each predicate type,
    the selected predicates and the classifier weights."""
    with open(glob.glob(os.path.join(out, "*.neupi_info"))[0], "rb") as f:
        info = pkl.load(f)
    vectors = {
        str(pred): _nested_lists(entry["ae_vecs"])
        for pred, entry in info["learned_ae_pred_info"].items()
    }
    selected = sorted((str(template), sorted(dummy.name
                                             for dummy, _, _ in variants))
                      for template, variants in info["selected_pred"].items())
    weights = {
        os.path.relpath(path, out): torch.load(path, weights_only=False)
        for path in sorted(glob.glob(os.path.join(out, "*", "*_model.pth")))
    }
    return vectors, selected, weights


def test_same_seed_same_predicates(tmp_path):
    """Two runs with the same seed and the mock LLM learn the same
    predicates, with identical classifier weights."""
    runs = []
    for name in ("first", "second"):
        out = tmp_path / name
        out.mkdir()
        _run(out, "satellites", 10, [])
        runs.append(_learned(str(out)))
    (vectors, selected, weights), (vectors2, selected2, weights2) = runs
    assert vectors == vectors2
    assert selected == selected2
    assert weights and weights.keys() == weights2.keys()
    for path, state in weights.items():
        for key, tensor in state.items():
            assert torch.equal(tensor, weights2[path][key]), (path, key)
