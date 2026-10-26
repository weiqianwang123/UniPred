"""The real-robot chain on the synthetic fixture with the fake perception
backends and the mock LLM: predicate invention, offline evaluation,
planning from an image and the closed loop; and the VLM baselines."""

import os
import subprocess
import sys

import pytest
import yaml

from predicators.approaches.bilevel_learning_approach import \
    read_predicate_config

from conftest import REPO

_TINY = dict(epochs=2, val_freq=1, num_iter=1, batch_vect_num=1,
             batch_size=32, guidance_thresh=1.0, loss_thresh=0.9)


def _main(flags):
    result = subprocess.run([sys.executable, "-m", "unipred.main"] + flags,
                            cwd=REPO,
                            capture_output=True,
                            text=True,
                            timeout=1800,
                            check=False)
    assert result.returncode == 0, result.stderr[-3000:]


def _flags(config, processed, out):
    return [
        "--config", config, "--config", "configs/llm/mock.yaml", "--seed",
        "0", "--table_clean_real_data_dir", processed, "--num_train_tasks",
        "4", "--num_test_tasks", "2", "--approach_dir",
        os.path.join(out, "approach"), "--neupi_save_path",
        os.path.join(out, "approach"), "--llm_cache_dir", ""
    ]


def test_unipred_chain(processed, tmp_path):
    """Invent predicates, evaluate offline, plan from an image and run the
    closed loop on a replayed session."""
    config = read_predicate_config("configs/table_clean_real/predicates.yaml")
    config["defaults"].update(_TINY)
    predicates = tmp_path / "predicates.yaml"
    predicates.write_text(yaml.safe_dump(config))
    flags = _flags("configs/table_clean_real/unipred.yaml", processed,
                   str(tmp_path)) + ["--neupi_pred_config", str(predicates)]
    log = tmp_path / "log.txt"
    _main(flags + [
        "--results_dir",
        str(tmp_path), "--eval_trajectories_dir",
        str(tmp_path), "--log_file",
        str(log)
    ])
    assert "Tasks solved:" in log.read_text()
    for script, session in (("plan_from_image.py", "test_00"),
                            ("closed_loop_example.py", "train_00")):
        result = subprocess.run([
            sys.executable, f"examples/real_robot/{script}", "--session",
            f"examples/real_robot/fixture/raw/{session}", "--detector",
            "fake", "--encoder", "fake"
        ] + flags,
                                cwd=REPO,
                                capture_output=True,
                                text=True,
                                timeout=1800,
                                check=False)
        assert result.returncode == 0, result.stderr[-3000:]
        assert "Plan" in result.stderr or "No plan" in result.stderr


@pytest.mark.parametrize("config", [
    "vila_zero_shot", "vila_few_shot", "vila_hpe", "pix2pred",
    "pix2pred_gt_predicates", "pix2pred_gt_labels"
])
def test_vlm_baselines(processed, tmp_path, config):
    """The Table III baselines run with the mock LLM and are scored by
    offline planning."""
    log = tmp_path / "log.txt"
    _main(
        _flags(f"configs/table_clean_real/{config}.yaml", processed,
               str(tmp_path)) + [
                   "--results_dir",
                   str(tmp_path), "--eval_trajectories_dir",
                   str(tmp_path), "--log_file",
                   str(log)
               ])
    assert "Tasks solved:" in log.read_text()
