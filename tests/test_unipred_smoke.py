"""End-to-end smoke tests: UniPred on a few tiny tasks of each domain with
the mock LLM (data, PDDL completion, LLM loop, selection, operator learning
and evaluation)."""

import glob
import json
import os
import re
import subprocess
import sys

import dill as pkl
import pytest
import torch
import yaml

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from unipred.config import load_config
from unipred.llm.names import NameMap

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TINY = dict(epochs=2, val_freq=1, num_iter=1, batch_vect_num=1,
             batch_size=32, guidance_thresh=1.0, loss_thresh=0.9)


def _run(tmp_path, domain, num_train_tasks, extra_flags, **learning):
    """Run UniPred with the tiny learning settings (updated by learning)
    and return the log."""
    with open(os.path.join(_REPO, f"configs/{domain}/predicates.yaml"),
              encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tiny = {**_TINY, **learning}
    config["defaults"].update(tiny)
    for entry in config["predicate_types"]:
        for key in tiny:
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
    assert "Tasks solved:" in log
    assert any(name.endswith(".llm_history.json") for name in os.listdir(out))
    return log


@pytest.mark.parametrize("domain, num_train_tasks", [
    ("satellites", 10),
    ("blocks", 10),
    ("table_clean_sim", 5),
])
def test_unipred_mock(tmp_path, domain, num_train_tasks):
    """UniPred runs end to end on each domain."""
    log = _run(tmp_path, domain, num_train_tasks, [])
    assert "Proposing PDDL seed" in log


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


def _llm_log(out):
    """The logged LLM requests of a run: (task, system, user, response)."""
    with open(glob.glob(os.path.join(out, "*.llm_log.jsonl"))[0],
              encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return [(r["task"], r["system"], r["user"], r["response"])
            for r in records]


def _neutral_names(domain):
    """Semantic PDDL symbol -> neutral id of the options and given
    predicates, as the w/o Semantic ablation names them."""
    excluded = load_config(os.path.join(
        _REPO, f"configs/{domain}/base.yaml"))["excluded_predicates"]
    utils.reset_config({"env": domain, "excluded_predicates": excluded})
    env = get_or_create_env(domain)
    options = get_gt_options(domain)
    predicates = {
        p
        for p in env.predicates if p.name not in excluded.split(",")
    }
    semantic = NameMap(options, predicates, semantic=True)
    neutral = NameMap(options, predicates, semantic=False)
    renames = {semantic.option(o): neutral.option(o) for o in options}
    renames.update({semantic.predicate(p): neutral.predicate(p)
                    for p in predicates})
    return renames


def _rename(text, renames):
    for name in sorted(renames, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", renames[name], text)
    return text


def test_ablations_change_only_what_they_name(tmp_path):
    """Each ablation overlay changes only its part of what the LLM is
    asked."""
    domain = "table_clean_sim"
    logs = {}
    for name in ("default", "no_semantic", "no_loss_feedback",
                 "no_pddl_completion", "no_llm_loop",
                 "no_derived_selection"):
        out = tmp_path / name
        out.mkdir()
        flags = [] if name == "default" else [
            "--config", f"configs/ablations/{name}.yaml"
        ]
        # A second iteration asks the LLM with feedback on trained
        # proposals.
        _run(out, domain, 5, flags, num_iter=2)
        logs[name] = _llm_log(str(out))
    default = logs["default"]
    tasks = [task for task, _, _, _ in default]
    assert tasks[0] == "pddl_completion" and "effect_proposal" in tasks
    assert any("-> score " in user for _, _, user, _ in default)
    # w/o Semantic: the same requests and replies, up to the names of the
    # options and given predicates, which appear nowhere.
    renames = _neutral_names(domain)
    assert [tuple(_rename(part, renames) for part in record)
            for record in default] == logs["no_semantic"]
    hidden = re.compile("|".join(rf"\b{re.escape(name)}\b"
                                 for name in renames))
    assert not any(
        hidden.search(part) for record in logs["no_semantic"]
        for part in record[1:])
    # w/o loss feedback: the same requests, with accepted/rejected instead
    # of scores.
    no_feedback = logs["no_loss_feedback"]
    assert no_feedback[0] == default[0]
    assert not any("score" in user for _, _, user, _ in no_feedback)
    assert any(" -> rejected" in user or " -> accepted" in user
               for _, _, user, _ in no_feedback)
    # w/o PDDL completion: only the LLM loop; w/o the loop: only PDDL
    # completion.
    assert {task for task, _, _, _ in logs["no_pddl_completion"]
            } == {"effect_proposal"}
    assert logs["no_llm_loop"] == default[:1]
    # w/o derived selection does not change what the LLM is asked.
    assert logs["no_derived_selection"] == default
