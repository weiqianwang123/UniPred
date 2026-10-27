"""Every YAML file under configs/ loads: experiment configs and overlays
expand into known flags, predicate configs name known options and types,
and the real-robot configs are complete."""

import glob
import os
import sys

import pytest
import yaml

from predicators import utils
from predicators.approaches.bilevel_learning_approach import \
    load_predicate_config
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from unipred.config import expand_config_args, load_config
from unipred.llm.pddl import load_known_predicates
from unipred.perception.scene import load_robot_config

from conftest import REPO

_CONFIGS = sorted(
    os.path.relpath(p, REPO)
    for p in glob.glob(os.path.join(REPO, "configs", "**", "*.yaml"),
                       recursive=True))
_PREDICATE_CONFIGS = [
    p for p in _CONFIGS
    if os.path.basename(p) in ("predicates.yaml", "ivntr_predicates.yaml")
]
_SPECIAL = {
    "configs/real_robot.yaml", "configs/sim_robot.yaml",
    "configs/table_clean_real/domain.yaml"
}
_EXPERIMENT_CONFIGS = [
    p for p in _CONFIGS if p not in _PREDICATE_CONFIGS and p not in _SPECIAL
]
# The simulated domain whose options and types a domain's configs use.
_SIM_ENV = {
    "satellites": "satellites",
    "blocks": "blocks_onclear",
    "packing": "packing",
    "table_clean_sim": "table_clean_sim",
    "table_clean_real": "table_clean_sim",
}


def test_all_configs_are_covered():
    """Every config is checked by one of the tests below."""
    assert len(_CONFIGS) == len(_PREDICATE_CONFIGS) + len(
        _EXPERIMENT_CONFIGS) + len(_SPECIAL)


@pytest.mark.parametrize("path", _EXPERIMENT_CONFIGS)
def test_experiment_config(path):
    """The config expands into flags that predicators knows, and the files
    it names exist."""
    os.chdir(REPO)
    argv = sys.argv
    sys.argv = ["main.py"] + expand_config_args(["--config", path])
    try:
        # As predicators' main does; raises for an unknown flag.
        utils.parse_args(env_required=False,
                         approach_required=False,
                         seed_required=False)
    finally:
        sys.argv = argv
    flags = load_config(path)
    for key in ("neupi_pred_config", "vila_constraints"):
        if flags.get(key):
            assert os.path.exists(flags[key]), f"{path}: no {flags[key]}"


@pytest.mark.parametrize("path", _PREDICATE_CONFIGS)
def test_predicate_config(path):
    """The predicate config names the options and types of its domain, and
    its known predicates exist with their effects."""
    env_name = _SIM_ENV[os.path.basename(os.path.dirname(path))]
    utils.reset_config({"env": env_name, "perception_encoder": "fake"})
    env = get_or_create_env(env_name)
    options = sorted(get_gt_options(env_name))
    full_path = os.path.join(REPO, path)
    predicate_types, _, static = load_predicate_config(full_path, options)
    type_names = {t.name for t in env.types}
    for entry in predicate_types:
        assert set(entry["types"]) <= type_names, (path, entry["name"])
    with open(full_path, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if "include" not in content:
        load_known_predicates(content.get("known_predicates", []),
                              env.predicates, options)
        assert set(static) <= {p.name for p in env.predicates}


def test_real_robot_configs():
    """The robot configs (the real robot's and the simulated one's) give a
    prompt per object type of the domain spec, whose predicates use its
    types."""
    os.chdir(REPO)
    with open("configs/table_clean_real/domain.yaml", encoding="utf-8") as f:
        domain = yaml.safe_load(f)
    for path in ("configs/real_robot.yaml", "configs/sim_robot.yaml"):
        robot = load_robot_config(path)
        assert set(domain["types"]) <= set(robot["prompts"])
        assert set(robot["max_box_area"]) <= set(robot["prompts"])
    for types in domain["predicates"].values():
        assert set(types) <= set(domain["types"])
    assert set(domain["goal_predicates"]) <= set(domain["predicates"])
    assert set(domain["derived_predicates"]) <= set(domain["predicates"])
