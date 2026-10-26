"""Tests for experiment configs."""

import pytest

from unipred.config import expand_config_args, load_config


def test_include_and_override(tmp_path):
    """Included files load first; the including file and explicit flags
    override them."""
    (tmp_path / "base.yaml").write_text("env: satellites\ntimeout: 5\n"
                                        "load_data: true\n")
    (tmp_path / "exp.yaml").write_text("include: base.yaml\ntimeout: 10\n"
                                       "excluded_predicates: A,B\n")
    assert load_config(str(tmp_path / "exp.yaml")) == {
        "env": "satellites",
        "timeout": 10,
        "load_data": True,
        "excluded_predicates": "A,B",
    }
    args = expand_config_args(
        ["--config", str(tmp_path / "exp.yaml"), "--seed", "3",
         "--timeout", "1"])
    assert args == [
        "--env", "satellites", "--timeout", "1", "--load_data",
        "--excluded_predicates", "A,B", "--seed", "3"
    ]


def test_store_true_false_and_lists(tmp_path):
    """False store_true flags are dropped and lists use predicators'
    syntax."""
    (tmp_path / "c.yaml").write_text("load_data: false\n"
                                     "mlp_classifier_hid_sizes: [32, 32]\n")
    assert expand_config_args(["--config", str(tmp_path / "c.yaml")]) == [
        "--mlp_classifier_hid_sizes", "[32,32]"
    ]


def test_bad_arguments(tmp_path):
    """Malformed arguments and values are errors."""
    with pytest.raises(ValueError):
        expand_config_args(["seed", "0"])
    with pytest.raises(ValueError):
        expand_config_args(["--seed"])
    (tmp_path / "c.yaml").write_text("timeout: {a: 1}\n")
    with pytest.raises(ValueError):
        expand_config_args(["--config", str(tmp_path / "c.yaml")])
