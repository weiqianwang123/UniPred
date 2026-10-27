"""Tests for the object features that learned samplers see."""

import numpy as np

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.nsrt_learning.sampler_learning import _sampler_features
from predicators.settings import CFG


def test_satellites_samplers_do_not_see_ids():
    """In Satellites, the id features are not sampler inputs; elsewhere all
    features are."""
    utils.reset_config({"env": "satellites"})
    feature_filter = CFG.domain_sampler_data_filter
    env = get_or_create_env("satellites")
    state = env.get_train_tasks()[0].init
    for obj in state:
        names = obj.type.feature_names
        kept = [names[i] for i in feature_filter[obj.type.name]]
        assert not {"id", "calibration_obj_id", "read_obj_id"} & set(kept)
        assert len(kept) == len(names) - (2 if obj.type.name == "satellite"
                                          else 1)
        assert np.array_equal(_sampler_features(state, obj, feature_filter),
                              [state.get(obj, n) for n in kept])
        assert np.array_equal(_sampler_features(state, obj, None),
                              state[obj])
    utils.reset_config({"env": "blocks"})
    assert CFG.domain_sampler_data_filter is None
