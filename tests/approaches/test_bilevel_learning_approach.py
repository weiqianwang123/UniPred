"""Tests for the extension points of the IVNTR approach."""

import os

import pytest
import torch

from predicators import utils
from predicators.approaches.bilevel_learning_approach import \
    BilevelLearningApproach, PredicateCandidate
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Predicate

_PREDS = {
    name: Predicate(name, [], lambda s, o: True)
    for name in ("P0", "A", "B", "C")
}


class _FakeScoreFunction:
    """Scores effect predicate sets from a table and records the calls."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def evaluate(self, matrix, predicates, ent_idx, strips_learner,
                 operator_predicates=None):
        """Look up the score of the effect predicates."""
        del strips_learner  # unused
        assert matrix.shape[1] == len(ent_idx)
        effects = frozenset(p.name for p in predicates[:len(ent_idx)])
        self.calls.append((effects, operator_predicates))
        return self._scores.get(effects, 1e9)


class _ExpandingApproach(BilevelLearningApproach):
    """Adds C to the pool once A is selected and learns operators from P0
    and A only."""

    def _expand_candidate_pool(self, selected, pool):
        if selected[-1] == _PREDS["A"]:
            pool.append(_candidate("C"))

    def _operator_predicates(self, predicates):
        return [p for p in predicates if p.name in ("P0", "A")]


def _candidate(name):
    return PredicateCandidate(_PREDS[name], torch.zeros((4, 2),
                                                        dtype=torch.long),
                              [0])


def _make_approach(tmp_path, approach_cls):
    config = tmp_path / "pred.yaml"
    config.write_text("final_op: []\nneupi_non_effect_predicates: []\n"
                      "config: []\n")
    utils.reset_config({
        "env": "blocks",
        "neupi_pred_config": str(config),
        "domain_aaai_thresh": 10,
    })
    env = get_or_create_env("blocks")
    return approach_cls(set(), get_gt_options("blocks"), env.types,
                        env.action_space, [])


def _select(approach, candidate_names, scores):
    """Run selection with P0 provided and the named candidates."""
    names = ["P0"] + candidate_names
    huge_matrix = torch.zeros((4, len(names), 2), dtype=torch.long)
    info = {
        i: (_PREDS[name], [0], name == "P0", (_PREDS[name], 0, 0))
        for i, name in enumerate(names)
    }
    score_function = _FakeScoreFunction(scores)
    _, matrix, selected, ent_idx = \
        approach._select_predicates_by_score_search(  # pylint: disable=protected-access
            huge_matrix, info, score_function)
    assert matrix.shape[1] == len(selected) == len(ent_idx)
    return [p.name for p in selected], score_function.calls


def test_selection_expands_candidate_pool(tmp_path):
    """Candidates added by _expand_candidate_pool are tried next, and
    _operator_predicates reaches the score function."""
    approach = _make_approach(tmp_path, _ExpandingApproach)
    selected, calls = _select(
        approach, ["A", "B"], {
            frozenset({"P0", "A"}): 50,
            frozenset({"P0", "B"}): 80,
            frozenset({"P0", "A", "B"}): 60,
            frozenset({"P0", "A", "C"}): 5,
        })
    assert selected == ["P0", "A", "C"]
    last_effects, operator_predicates = calls[-1]
    assert last_effects == {"P0", "A", "C"}
    assert {p.name for p in operator_predicates} == {"P0", "A"}


def test_selection_stops_when_nothing_is_left_to_imagine(tmp_path):
    """A stuck search imagines the remaining candidates, takes them back
    and stops instead of looping forever."""
    approach = _make_approach(tmp_path, BilevelLearningApproach)
    selected, calls = _select(
        approach, ["A", "B"], {
            frozenset({"P0", "A"}): 50,
            frozenset({"P0", "B"}): 80,
            frozenset({"P0", "A", "B"}): 70,
        })
    assert selected == ["P0", "A"]
    assert len(calls) == 3


def test_selection_without_candidates(tmp_path):
    """With no candidates, only the provided predicates are selected."""
    approach = _make_approach(tmp_path, BilevelLearningApproach)
    selected, calls = _select(approach, [], {})
    assert selected == ["P0"]
    assert not calls


def test_model_artifact_path():
    """Artifact paths only change the file name."""
    artifact_path = BilevelLearningApproach._model_artifact_path  # pylint: disable=protected-access
    model_dir = os.path.join("saved_model", "neural_model_p1")
    model = os.path.join(model_dir, "iter_0_1_model.pth")
    assert artifact_path(model, "guidance") == os.path.join(
        model_dir, "iter_0_1_guidance.pth")
    good = artifact_path(model, "model_good")
    assert good == os.path.join(model_dir, "iter_0_1_model_good.pth")
    assert artifact_path(good, "ae_vector") == os.path.join(
        model_dir, "iter_0_1_ae_vector_good.pth")
    with pytest.raises(AssertionError):
        artifact_path(os.path.join(model_dir, "weights.pth"), "guidance")
