"""Tests for the extension points of the IVNTR approach."""

import os

import pytest
import torch

from predicators import hooks, utils
from predicators.approaches.bilevel_learning_approach import \
    BilevelLearningApproach, PredicateCandidate, load_predicate_config
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

    def evaluate(self,
                 matrix,
                 predicates,
                 ent_idx,
                 strips_learner,
                 operator_predicates=None,
                 atoms_closure=None):
        """Look up the score of the effect predicates."""
        del strips_learner  # unused
        assert matrix.shape[1] == len(ent_idx)
        effects = frozenset(p.name for p in predicates[:len(ent_idx)])
        self.calls.append((effects, operator_predicates, atoms_closure))
        return self._scores.get(effects, 1e9)


# A stand-in closure; selection only passes it on.
_CLOSURE_FOR_C = hooks.AtomsClosure(frozenset(), lambda atoms, objects: atoms)


class _ExpandingApproach(BilevelLearningApproach):
    """Adds C to the pool once A is selected, learns operators from P0 and A
    only and treats C as derived."""

    def _expand_candidate_pool(self, selected, pool):
        if selected[-1] == _PREDS["A"]:
            pool.append(_candidate("C"))

    def _operator_predicates(self, predicates):
        return [p for p in predicates if p.name in ("P0", "A")]

    def _atoms_closure(self, predicates):
        if _PREDS["C"] in predicates:
            return _CLOSURE_FOR_C
        return None


def _candidate(name):
    return PredicateCandidate(_PREDS[name], torch.zeros((4, 2),
                                                        dtype=torch.long),
                              [0])


def _make_approach(tmp_path, approach_cls):
    config = tmp_path / "pred.yaml"
    config.write_text("predicate_types: []\n")
    utils.reset_config({
        "env": "blocks",
        "neupi_pred_config": str(config),
        "neupi_selection_thresh": 10,
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
    _operator_predicates and _atoms_closure reach the score function."""
    approach = _make_approach(tmp_path, _ExpandingApproach)
    selected, calls = _select(
        approach, ["A", "B"], {
            frozenset({"P0", "A"}): 50,
            frozenset({"P0", "B"}): 80,
            frozenset({"P0", "A", "B"}): 60,
            frozenset({"P0", "A", "C"}): 5,
        })
    assert selected == ["P0", "A", "C"]
    last_effects, operator_predicates, closure = calls[-1]
    assert last_effects == {"P0", "A", "C"}
    assert {p.name for p in operator_predicates} == {"P0", "A"}
    assert closure is _CLOSURE_FOR_C
    assert all(call[2] is None for call in calls if "C" not in call[0])


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


def test_load_predicate_config(tmp_path):
    """Defaults are merged into every predicate type, final options are
    named, and unknown option names are errors."""
    utils.reset_config({"env": "blocks"})
    options = sorted(get_gt_options("blocks"), key=lambda o: o.name)
    names = [o.name for o in options]
    path = tmp_path / "pred.yaml"
    path.write_text(f"final_options: [{names[1]}]\n"
                    "static_predicates: [Given]\n"
                    "defaults: {epochs: 3, gumbel_temp: 0.5}\n"
                    "predicate_types:\n"
                    "  - {name: p, types: [block], epochs: 7}\n"
                    f"  - {{name: q, types: [block], "
                    f"search_options: [{names[0]}]}}\n")
    types, is_final, static = load_predicate_config(str(path), options)
    assert [t["epochs"] for t in types] == [7, 3]
    assert all(t["gumbel_temp"] == 0.5 and t["decision_b"] == 0.5
               for t in types)
    assert is_final == [int(i == 1) for i in range(len(options))]
    assert static == ["Given"]
    path.write_text("final_options: [Fly]\npredicate_types: []\n")
    with pytest.raises(ValueError, match="unknown options"):
        load_predicate_config(str(path), options)


def test_predicate_config_include(tmp_path):
    """An included config is read first; defaults and same-named types are
    updated, other types are kept."""
    utils.reset_config({"env": "blocks"})
    options = sorted(get_gt_options("blocks"), key=lambda o: o.name)
    (tmp_path / "base.yaml").write_text(
        "description: base\n"
        "defaults: {epochs: 3, num_iter: 5}\n"
        "predicate_types:\n"
        "  - {name: p, types: [block], num_iter: 1}\n"
        "  - {name: q, types: [block]}\n")
    (tmp_path / "pred.yaml").write_text(
        "include: base.yaml\n"
        "defaults: {num_iter: 70}\n"
        "predicate_types:\n"
        "  - {name: p, epochs: 9}\n")
    types, _, _ = load_predicate_config(str(tmp_path / "pred.yaml"), options)
    assert [(t["name"], t["epochs"], t["num_iter"]) for t in types] == \
        [("p", 9, 1), ("q", 3, 70)]
