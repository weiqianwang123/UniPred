"""Tests for the hooks of environments built from recorded data."""

import pytest

from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv
from predicators.envs.satellites import SatellitesEnv
from predicators.structs import Dataset


def test_recorded_dataset_and_offline_evaluation(monkeypatch):
    """With --offline_data_method recorded, the dataset comes from the
    environment; by default there is none, and test tasks are executed."""
    utils.reset_config({
        "env": "satellites",
        "offline_data_method": "recorded"
    })
    env = SatellitesEnv()
    with pytest.raises(NotImplementedError):
        create_dataset(env, [], set())
    dataset = Dataset([])
    monkeypatch.setattr(env, "get_recorded_dataset", lambda: dataset)
    assert create_dataset(env, [], set()) is dataset
    assert BaseEnv.evaluate_offline(env, None) is None
