"""Tests for derived rules and the derived closure."""

import pytest

from predicators.structs import GroundAtom, Predicate, Type
from unipred.derived.closure import make_closure
from unipred.derived.rules import DerivedRule

_TOY = Type("toy", [])
_TABLE = Type("table", [])
_ON_TABLE = Predicate("OnTable", [_TOY, _TABLE], lambda s, o: True)
_TABLE_CLEAR = Predicate("TableClear", [_TABLE], lambda s, o: True)
_SOME_ON = Predicate("SomeToyOn", [_TABLE], lambda s, o: True)
_NOT_ON = Predicate("NotOnTable", [_TOY, _TABLE], lambda s, o: True)
_ANY_ON = Predicate("AnyToyOnAnyTable", [], lambda s, o: True)

_T0, _T1 = _TOY("t0"), _TOY("t1")
_TAB = _TABLE("tab")
_OBJECTS = {_T0, _T1, _TAB}

_FORALL_NOT = DerivedRule(_TABLE_CLEAR, _ON_TABLE, negated=True,
                          quantifier="ForAll", quantified=(0, ))
_EXISTS = DerivedRule(_SOME_ON, _ON_TABLE, quantifier="Exists",
                      quantified=(0, ))
_NOT = DerivedRule(_NOT_ON, _ON_TABLE, negated=True)


def test_quantified_and_negated_rules():
    """ForAll/Exists quantify over the objects of the quantified type and the
    negation applies inside the quantifier."""
    one_on = {GroundAtom(_ON_TABLE, [_T0, _TAB])}
    assert _FORALL_NOT.evaluate(one_on, _OBJECTS) == set()
    assert _FORALL_NOT.evaluate(set(), _OBJECTS) == {
        GroundAtom(_TABLE_CLEAR, [_TAB])
    }
    assert _EXISTS.evaluate(one_on, _OBJECTS) == {
        GroundAtom(_SOME_ON, [_TAB])
    }
    assert _NOT.evaluate(one_on, _OBJECTS) == {
        GroundAtom(_NOT_ON, [_T1, _TAB])
    }
    both = DerivedRule(_ANY_ON, _ON_TABLE, quantifier="Exists",
                       quantified=(0, 1))
    assert both.evaluate(one_on, _OBJECTS) == {GroundAtom(_ANY_ON, [])}


def test_empty_domain_and_repeated_objects():
    """As in IVNTR's classifiers, a quantifier over no groundings is false,
    and groundings that repeat an object are skipped."""
    assert _FORALL_NOT.evaluate(set(), {_TAB}) == set()
    _BLOCK = Type("block", [])
    on = Predicate("On", [_BLOCK, _BLOCK], lambda s, o: True)
    clear = Predicate("Clear", [_BLOCK], lambda s, o: True)
    rule = DerivedRule(clear, on, negated=True, quantifier="ForAll",
                       quantified=(0, ))
    b0, b1 = _BLOCK("b0"), _BLOCK("b1")
    assert rule.evaluate({GroundAtom(on, [b1, b0])}, {b0, b1}) == {
        GroundAtom(clear, [b1])
    }


def test_rule_checks_head_types():
    """The head must take the free arguments of the base."""
    with pytest.raises(AssertionError):
        DerivedRule(_NOT_ON, _ON_TABLE, quantifier="ForAll", quantified=(0, ))
    with pytest.raises(AssertionError):
        DerivedRule(_NOT_ON, _ON_TABLE)


def test_closure_removes_stale_derived_atoms():
    """Derived atoms are recomputed from scratch, so an atom that no longer
    holds is removed."""
    closure = make_closure([_FORALL_NOT, _EXISTS])
    stale = {GroundAtom(_TABLE_CLEAR, [_TAB]),
             GroundAtom(_ON_TABLE, [_T1, _TAB])}
    assert closure(stale, _OBJECTS) == {
        GroundAtom(_ON_TABLE, [_T1, _TAB]),
        GroundAtom(_SOME_ON, [_TAB])
    }
    assert closure(set(), _OBJECTS) == {GroundAtom(_TABLE_CLEAR, [_TAB])}
