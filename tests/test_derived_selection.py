"""Tests for derived-aware predicate selection."""

import torch

from predicators import utils
from predicators.approaches.bilevel_learning_approach import \
    BilevelLearningApproach
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import DummyPredicate, GroundAtom, Predicate
from unipred.derived.selection import DerivedSelectionMixin


class _Derived(DerivedSelectionMixin, BilevelLearningApproach):
    """Derived-aware selection on top of IVNTR."""

    @classmethod
    def get_name(cls):
        return "test_derived"


class _RecordingScore:
    """Scores effect predicate sets from a table and records the calls."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def evaluate(self, matrix, predicates, ent_idx, strips_learner,
                 operator_predicates=None, atoms_closure=None):
        """Look up the score of the effect predicates."""
        del matrix, strips_learner  # unused
        effects = frozenset(p.name for p in predicates[:len(ent_idx)])
        self.calls.append((effects, operator_predicates, atoms_closure))
        return self._scores.get(effects, 1e9)


def test_derived_forms_enter_with_their_basic_predicate(tmp_path):
    """The derived form of B is tried only after B is selected; it is never
    an operator predicate, and its closure is used while scoring."""
    config = tmp_path / "pred.yaml"
    config.write_text("predicate_types: []\n")
    utils.reset_config({
        "env": "blocks",
        "neupi_pred_config": str(config),
        "neupi_selection_thresh": 10,
    })
    env = get_or_create_env("blocks")
    approach = _Derived(set(), get_gt_options("blocks"), env.types,
                        env.action_space, [])
    block = next(t for t in env.types if t.name == "block")
    template = DummyPredicate("tmpl", [block, block])
    approach.learned_ae_pred_info = {
        template: {
            "quantifiers": [[("", "", "tmpl_0(?blo0,?blo1)"),
                             ("ForAll:?blo", "Not", "tmpl_0(?blo,?blo1)")]]
        }
    }
    given = Predicate("Given", [], lambda s, o: True)
    base = Predicate("tmpl_0_blo0_blo1", [block, block], lambda s, o: True)
    derived = Predicate("forall_blo_Not_tmpl_0_blo_blo1", [block],
                        lambda s, o: True)
    other = Predicate("Other", [block], lambda s, o: True)
    info = {
        0: (given, [], True, (given, 0, 0)),
        1: (base, [0, 1], False, (template, 0, 0)),
        2: (derived, [1], False, (template, 0, 1)),
        3: (other, [0], False, (other, 0, 0)),
    }
    score = _RecordingScore({
        frozenset({"Given", base.name}): 50,
        frozenset({"Given", "Other"}): 80,
        frozenset({"Given", base.name, derived.name}): 5,
    })
    matrix = torch.zeros((len(approach.ae_row_names), 4, 2), dtype=torch.long)
    _, _, selected, _ = approach._select_predicates_by_score_search(  # pylint: disable=protected-access
        matrix, info, score)
    assert [p.name for p in selected] == ["Given", base.name, derived.name]
    # The first level only tries the basic candidates.
    first_level = {effects for effects, _, _ in score.calls[:2]}
    assert first_level == {
        frozenset({"Given", base.name}),
        frozenset({"Given", "Other"})
    }
    effects, operator_predicates, closure = score.calls[-1]
    assert derived.name in effects
    assert derived not in operator_predicates
    b0, b1 = block("b0"), block("b1")
    atoms = closure({GroundAtom(base, [b0, b1])}, {b0, b1})
    # ForAll x. not tmpl(x, b0) holds; for b1, x = b0 makes tmpl true.
    assert GroundAtom(derived, [b0]) in atoms
    assert GroundAtom(derived, [b1]) not in atoms
    rule = approach._derived_rules[derived]  # pylint: disable=protected-access
    assert (rule.base, rule.negated, rule.quantifier, rule.quantified) == \
        (base, True, "ForAll", (0, ))
