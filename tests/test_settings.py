"""Tests for registering extension settings."""

import sys

import pytest

from predicators import utils
from predicators.settings import CFG, GlobalSettings, \
    register_arg_specific_settings, register_settings


class _ExtensionSettings:
    """Settings of a fake extension."""
    ext_test_value = 3
    ext_test_name = "default"


def test_register_settings(monkeypatch):
    """Registered settings behave like built-in ones."""
    register_settings(_ExtensionSettings)
    assert GlobalSettings.ext_test_value == 3
    assert CFG.ext_test_value == 3
    # Registering the same class again is a no-op.
    register_settings(_ExtensionSettings)
    utils.reset_config({"env": "satellites", "ext_test_value": 5})
    assert CFG.ext_test_value == 5
    utils.reset_config({"env": "satellites"})
    assert CFG.ext_test_value == 3
    assert CFG.ext_test_name == "default"
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--env", "satellites", "--approach", "oracle", "--seed",
        "0", "--ext_test_value", "7"
    ])
    assert utils.parse_args()["ext_test_value"] == 7


@pytest.mark.parametrize("clashing_name",
                         ["num_train_tasks", "timeout", "ext_test_value"])
def test_register_settings_name_clash(clashing_name):
    """Names of settings, command line args and registered settings are
    taken."""
    register_settings(_ExtensionSettings)
    clashing_cls = type("_Clashing", (), {clashing_name: 1})
    with pytest.raises(ValueError, match="name is taken"):
        register_settings(clashing_cls)


def test_register_arg_specific_settings():
    """Registered arg-specific settings depend on the args and can be
    overridden from the command line."""

    def _settings(args):
        return {"ext_test_per_env": 2 if args.get("env") == "blocks" else 1}

    register_arg_specific_settings(_settings)
    utils.reset_config({"env": "blocks"})
    assert CFG.ext_test_per_env == 2
    utils.reset_config({"env": "satellites"})
    assert CFG.ext_test_per_env == 1
    utils.reset_config({"env": "satellites", "ext_test_per_env": 4})
    assert CFG.ext_test_per_env == 4
