"""Tests for LLM effect proposals."""

import numpy as np
import pytest

from predicators import utils
from predicators.structs import Action, Predicate, Type
from unipred.llm.client import LLMClient
from unipred.llm.effect_proposer import EffectProposer, allowed_codes, \
    loss_to_score
from unipred.llm.names import NameMap
from unipred.llm.pddl import ADD, DELETE, NO_EFFECT

_ROBOT = Type("robot", [])
_TOY = Type("toy", [])
_HOLDING = Predicate("neural_b_robot_toy", [_ROBOT, _TOY], lambda s, o: True)
_HAND_EMPTY = Predicate("HandEmpty", [_ROBOT], lambda s, o: True)


def _option(name):
    return utils.SingletonParameterizedOption(
        name, lambda s, m, o, p: Action(np.zeros(1)), types=[_ROBOT, _TOY])


_ROWS = [_option("Pick"), _option("Place"), _option("Wipe")]


class _Scripted(LLMClient):
    """Replies with the given texts in order and records the prompts."""

    def __init__(self, replies):
        super().__init__("mock", "mock", 0.2, 100, 10, 3)
        self._replies = list(replies)
        self.prompts = []

    def complete(self, system, user, task, attempt=0):
        self.prompts.append((system, user))
        return self._replies.pop(0)


def _proposer(client, seeds=(), use_llm=True, feedback=True, semantic=True,
              allowed=None):
    return EffectProposer(
        client=client, target=_HOLDING, rows=_ROWS,
        known_predicates=[_HAND_EMPTY],
        names=NameMap(_ROWS, [_HAND_EMPTY], semantic),
        allowed=allowed or [{0, 1, 2}] * 3, seeds=seeds, use_llm=use_llm,
        loss_feedback=feedback, description="Toys on a table.",
        accept_loss=0.005, zero_loss=1.0)


def test_loss_to_score():
    """Scores are monotonic, 100 at the acceptance threshold and 0 at the
    zero loss or for infeasible proposals."""
    assert loss_to_score(0.001, 0.005, 1.0) == 100
    assert loss_to_score(0.005, 0.005, 1.0) == 100
    assert loss_to_score(1.0, 0.005, 1.0) == 0
    assert loss_to_score(5.0, 0.005, 1.0) == 0
    assert loss_to_score(float("inf"), 0.005, 1.0) == 0
    assert 0 < loss_to_score(0.2, 0.005, 1.0) < loss_to_score(
        0.02, 0.005, 1.0) < 100


def test_allowed_codes():
    """Position constraints remove or force effect codes."""
    rules = [("position", 0, 0, 0, False), ("position", 1, 0, 1, True)]
    assert allowed_codes(3, rules) == [{NO_EFFECT, DELETE}, {DELETE},
                                       {NO_EFFECT, ADD, DELETE}]


def test_seeds_then_llm_with_feedback():
    """Seeds come first; the LLM then sees earlier proposals with scores,
    invalid replies are retried with a note, and None ends the search."""
    client = _Scripted([
        "ADD: Pick\nDEL: Place\nWHY: seed again",  # already tried
        "ADD: Wipe, Unknown\nDEL:\nWHY: bad name",
        "ADD:\nDEL: Wipe\nWHY: ok",
        "nonsense", "nonsense", "nonsense"
    ])
    proposer = _proposer(client, seeds=[np.array([ADD, DELETE, 0])])
    assert list(proposer.propose()) == [ADD, DELETE, NO_EFFECT]
    proposer.record_losses([np.array([ADD, DELETE, 0])], [0.002])
    assert list(proposer.propose()) == [NO_EFFECT, NO_EFFECT, DELETE]
    system, user = client.prompts[-1]
    assert "neural_b_robot_toy(robot, toy)" in system
    assert "Toys on a table." in system
    assert "ADD: pick | DEL: place -> score 100" in user
    assert "was already tried" in user and "Unknown is not an action" in user
    proposer.update_value(np.array([0, 0, DELETE]), np.array([np.inf]))
    assert proposer.propose() is None
    _, user = client.prompts[-1]
    assert "ADD:  | DEL: wipe -> impossible" in user


def test_without_feedback_or_semantics():
    """Without loss feedback there are no scores; without semantic names the
    options and known predicates get neutral ids."""
    client = _Scripted(["ADD: action0\nDEL:\nWHY: x"])
    proposer = _proposer(client, seeds=[np.array([0, ADD, 0])],
                         feedback=False, semantic=False)
    proposer.propose()
    proposer.record_losses([np.array([0, ADD, 0])], [0.3])
    assert list(proposer.propose()) == [ADD, NO_EFFECT, NO_EFFECT]
    system, user = client.prompts[-1]
    assert "score" not in user
    assert "Pick" not in system and "pick" not in system
    assert "HandEmpty" not in system and "pred0(robot)" in system
    assert "Toys on a table." in system


def test_seeds_only_and_forced_rows():
    """With the LLM loop off, only seeds are proposed; rows whose effect is
    forced by the data are filled in."""
    proposer = _proposer(None, seeds=[np.array([ADD, 0, 0])], use_llm=False)
    assert proposer.propose() is not None
    assert proposer.propose() is None
    client = _Scripted(["ADD: Pick\nDEL:\nWHY: x"])
    forced = _proposer(client, allowed=[{0, 1, 2}, {DELETE}, {0}])
    assert list(forced.propose()) == [ADD, DELETE, NO_EFFECT]
    with pytest.raises(AssertionError):
        _proposer(None, use_llm=True)
