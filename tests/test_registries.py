"""Tests for the subclass registries (envs, approaches, ground truth
models)."""

import sys

import dill
import pytest

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.envs import BaseEnv, get_or_create_env
from predicators.ground_truth_models import get_dummy_nsrts, get_gt_nsrts, \
    get_gt_options

_ENV_NAMES = ["satellites", "blocks", "blocks_onclear"]


def _registered_names(base_cls):
    return {
        cls.get_name()
        for cls in utils.get_all_subclasses(base_cls)
        if not cls.__abstractmethods__
    }


def test_registered_names():
    """The fork registers exactly these envs and approaches."""
    assert _registered_names(BaseEnv) == set(_ENV_NAMES)
    assert _registered_names(BaseApproach) == {
        "gnn_nsrt_policy", "grammar_search_invention", "ivntr",
        "nsrt_learning", "oracle", "random_nsrt", "random_options",
        "transformer_nsrt_policy"
    }


@pytest.mark.parametrize("env_name", _ENV_NAMES)
def test_ground_truth_models(env_name):
    """Every env gets its options and NSRTs from its registered
    factories."""
    utils.reset_config({"env": env_name})
    options = get_gt_options(env_name)
    assert options
    predicates = get_or_create_env(env_name).predicates
    nsrts = get_gt_nsrts(env_name, predicates, options)
    assert nsrts
    assert {nsrt.option for nsrt in nsrts}.issubset(options)


def test_options_survive_pickling():
    """Saved datasets and approaches contain options and their parameter
    spaces; they must load again (gym's spaces did not with NumPy >=
    1.25)."""
    utils.reset_config({"env": "satellites"})
    for option in get_gt_options("satellites"):
        loaded = dill.loads(dill.dumps(option))
        assert loaded.name == option.name
        assert loaded.params_space.contains(option.params_space.sample())


@pytest.mark.parametrize("env_name", ["satellites", "blocks_onclear"])
def test_dummy_nsrts(env_name):
    """Dummy NSRTs are registered for the IVNTR domains."""
    utils.reset_config({"env": env_name})
    options = get_gt_options(env_name)
    predicates = get_or_create_env(env_name).predicates
    assert get_dummy_nsrts(env_name, predicates, options)


def test_unknown_env_raises():
    """Envs without a registered factory fail loudly."""
    utils.reset_config({"env": "satellites"})
    with pytest.raises(NotImplementedError):
        get_gt_options("not_a_real_env")


def test_import_submodules_does_not_swallow_errors(tmp_path, monkeypatch):
    """A submodule with a missing dependency raises instead of silently
    dropping out of a registry."""
    pkg = tmp_path / "fake_registry_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text("VALUE = 1\n")
    (pkg / "bad.py").write_text("import a_module_that_does_not_exist\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ModuleNotFoundError):
            utils.import_submodules([str(pkg)], "fake_registry_pkg")
    finally:
        for name in list(sys.modules):
            if name.startswith("fake_registry_pkg"):
                del sys.modules[name]
