"""Smoke tests of the baselines: each runs end to end (data, learning,
testing) on a few small tasks."""

import os
import subprocess
import sys

import pytest
import yaml

from predicators.approaches.bilevel_learning_approach import \
    read_predicate_config

from conftest import REPO

_TINY = dict(epochs=2, val_freq=1, num_iter=1, batch_vect_num=1,
             batch_size=32, guidance_thresh=1.0, loss_thresh=0.9,
             matrix_vec_try=5)
# Small budgets so that every baseline finishes in seconds.
_SMALL = [
    "--gnn_num_epochs", "5", "--sampler_mlp_classifier_max_itr", "100",
    "--neural_gaus_regressor_max_itr", "100", "--mlp_regressor_max_itr",
    "100", "--grammar_search_max_predicates", "5",
    "--grammar_search_max_demos", "2"
]


@pytest.mark.parametrize("domain, config", [
    ("satellites", "random"),
    ("satellites", "gnn"),
    ("satellites", "transformer"),
    ("satellites", "grammar"),
    ("satellites", "ivntr"),
    ("table_clean_sim", "ivntr_derived"),
])
def test_baseline(tmp_path, domain, config):
    """The baseline learns from a few demonstrations and is tested."""
    out = str(tmp_path)
    flags = []
    if config.startswith("ivntr"):
        predicates = read_predicate_config(
            os.path.join(REPO, f"configs/{domain}/ivntr_predicates.yaml"))
        predicates["defaults"].update(_TINY)
        for entry in predicates["predicate_types"]:
            for key in _TINY:
                entry.pop(key, None)
        path = tmp_path / "predicates.yaml"
        path.write_text(yaml.safe_dump(predicates))
        flags = ["--neupi_pred_config", str(path)]
    command = [
        sys.executable, "-m", "unipred.main", "--config",
        f"configs/{domain}/{config}.yaml", "--seed", "0",
        "--num_train_tasks", "10", "--num_test_tasks", "1", "--approach_dir",
        out, "--neupi_save_path", out, "--data_dir", out, "--results_dir",
        out, "--eval_trajectories_dir", out, "--log_file",
        os.path.join(out, "log.txt")
    ] + _SMALL + flags
    result = subprocess.run(command,
                            cwd=REPO,
                            capture_output=True,
                            text=True,
                            timeout=1200,
                            check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Tasks solved:" in (tmp_path / "log.txt").read_text()
