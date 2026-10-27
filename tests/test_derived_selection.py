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


def _setup(tmp_path):
    """An approach, and the selection info of a provided predicate, a basic
    predicate with a derived form, and another candidate."""
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
    return approach, info, block, base, derived


def test_derived_forms_enter_with_their_basic_predicate(tmp_path):
    """The derived form of B is tried only with B, or after B is selected;
    it is never an operator predicate, and its closure is used while
    scoring."""
    approach, info, block, base, derived = _setup(tmp_path)
    score = _RecordingScore({
        frozenset({"Given", base.name}): 50,
        frozenset({"Given", "Other"}): 80,
        frozenset({"Given", base.name, derived.name}): 5,
    })
    matrix = torch.zeros((len(approach.ae_row_names), 4, 2), dtype=torch.long)
    _, _, selected, _ = approach._select_predicates_by_score_search(  # pylint: disable=protected-access
        matrix, info, score)
    assert [p.name for p in selected] == ["Given", base.name, derived.name]
    # The first level tries the basic candidates, B also with its derived
    # form, never the derived form alone.
    first_level = {effects for effects, _, _ in score.calls[:3]}
    assert first_level == {
        frozenset({"Given", base.name}),
        frozenset({"Given", base.name, derived.name}),
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


def test_basic_predicate_selected_for_its_derived_form(tmp_path):
    """A basic predicate that does not help alone is selected with the
    derived form that does; selection goes on below the threshold while the
    score improves by the refinement fraction."""
    approach, info, _, base, derived = _setup(tmp_path)
    matrix = torch.zeros((len(approach.ae_row_names), 4, 2), dtype=torch.long)
    score = _RecordingScore({
        frozenset({"Given", "Other"}): 8,
        frozenset({"Given", base.name}): 9,
        frozenset({"Given", base.name, derived.name}): 7,
        frozenset({"Given", base.name, derived.name, "Other"}): 6.99,
    })
    _, _, selected, _ = approach._select_predicates_by_score_search(  # pylint: disable=protected-access
        matrix, info, score)
    # Below the threshold (10) already, but the bundle lowers the score;
    # Other then improves it by less than 1 percent.
    assert [p.name for p in selected] == ["Given", base.name, derived.name]



def test_derived_preconditions_only_about_parameter_types():
    """A derived precondition whose basic predicate is about none of an
    operator's parameter types is dropped from it."""
    # pylint: disable=import-outside-toplevel
    from predicators.structs import NSRT, LiftedAtom, Type, Variable
    from predicators.utils import null_sampler
    from unipred.derived.rules import DerivedRule
    from unipred.derived.selection import _relevant_preconditions
    robot, wiper, toy, table = (Type(n, ["x"])
                                for n in ("robot", "wiper", "toy", "table"))
    on = Predicate("On", [toy, table], lambda s, o: True)
    no_toy_on = Predicate("NoToyOnAny", [], lambda s, o: True)
    holds = Predicate("Holds", [robot, toy], lambda s, o: True)
    free = Predicate("HoldsNoToy", [robot], lambda s, o: True)
    rules = {
        no_toy_on: DerivedRule(no_toy_on, on, True, "ForAll", (0, 1)),
        free: DerivedRule(free, holds, True, "ForAll", (1, )),
    }
    r, w = Variable("?r", robot), Variable("?w", wiper)
    option = utils.SingletonParameterizedOption(
        "PickWiper", lambda s, m, o, p: None, types=[robot, wiper])
    nsrt = NSRT("PickWiper", [r, w],
                {LiftedAtom(no_toy_on, []),
                 LiftedAtom(free, [r])}, set(), set(), set(), option, [r, w],
                null_sampler)
    kept = _relevant_preconditions(nsrt, rules)
    assert kept.preconditions == {LiftedAtom(free, [r])}


def test_derived_preconditions_kept_if_the_score_changes_without():
    """A derived precondition is dropped only if the score is the same
    without it (within 1 percent either way): one whose removal makes
    planning on the demonstrations cheaper rules plans out and is kept."""
    # pylint: disable=import-outside-toplevel
    from predicators.structs import NSRT, LiftedAtom, Type, Variable
    from predicators.utils import null_sampler
    from unipred.derived.selection import _prune_unneeded_preconditions
    robot, table = Type("robot", ["x"]), Type("table", ["x"])
    no_toy = Predicate("NoToyOn", [table], lambda s, o: True)
    free = Predicate("HoldsNoToy", [robot], lambda s, o: True)
    given = Predicate("Given", [robot], lambda s, o: True)
    r, t = Variable("?r", robot), Variable("?t", table)
    option = utils.SingletonParameterizedOption(
        "Wipe", lambda s, m, o, p: None, types=[robot, table])
    atoms = {LiftedAtom(no_toy, [t]), LiftedAtom(free, [r]),
             LiftedAtom(given, [r])}
    nsrt = NSRT("Wipe", [r, t], atoms, set(), set(), set(), option, [r, t],
                null_sampler)
    # Without NoToyOn wiping may come first: 1.4 percent cheaper. Without
    # HoldsNoToy nothing changes.
    costs = {"NoToyOn": 1.4, "HoldsNoToy": 0.0}

    def _score(nsrts):
        missing = {a.predicate.name for a in atoms} - \
            {a.predicate.name for a in nsrts[0].preconditions}
        return 100 - sum(costs[name] for name in missing)

    pruned, = _prune_unneeded_preconditions([nsrt], {no_toy, free}, _score)
    assert pruned.preconditions == {LiftedAtom(no_toy, [t]),
                                    LiftedAtom(given, [r])}
    costs["NoToyOn"] = 0.5
    pruned, = _prune_unneeded_preconditions([nsrt], {no_toy, free}, _score)
    assert pruned.preconditions == {LiftedAtom(given, [r])}
