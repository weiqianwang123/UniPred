"""Tests for PDDL completion."""

import json

import numpy as np
import pytest

from predicators import utils
from predicators.structs import Action, Predicate, Type
from unipred.llm.client import LLMClient
from unipred.llm.names import NameMap
from unipred.llm.pddl import ADD, DELETE, build_partial_domain, \
    complete_domain, extract_proposals, load_known_predicates, parse_domain, \
    proposals_to_seeds

_ROBOT = Type("robot", [])
_TOY = Type("toy", [])
_IN_BOX = Predicate("InBox", [_TOY], lambda s, o: True)
_HAND_EMPTY = Predicate("HandEmpty", [_ROBOT], lambda s, o: True)


def _option(name, types):
    return utils.SingletonParameterizedOption(
        name, lambda s, m, o, p: Action(np.zeros(1)), types=types)


_PICK = _option("PickToy", [_ROBOT, _TOY])
_PLACE = _option("PlaceToy", [_ROBOT, _TOY])
_OPTIONS = [_PICK, _PLACE]
_PREDICATES = {_IN_BOX, _HAND_EMPTY}


def _hints(entries):
    return load_known_predicates(entries, _PREDICATES, _OPTIONS)


def test_partial_domain():
    """Known effects and preconditions are bound to the option arguments by
    type."""
    hints = _hints([{
        "name": "InBox",
        "effects": {"PlaceToy": "add"}
    }, {
        "name": "HandEmpty",
        "precondition_of": ["PickToy"],
        "effects": {"PickToy": "del", "PlaceToy": "add"}
    }])
    names = NameMap(_OPTIONS, _PREDICATES, semantic=True)
    domain = build_partial_domain(_OPTIONS, _PREDICATES, hints, names)
    assert "(inbox ?toy0 - toy)" in domain
    assert ":precondition (and (handempty ?robot0))" in domain
    assert ":effect (and (not (handempty ?robot0)))" in domain
    parsed = parse_domain(domain)
    assert parsed.actions["placetoy"].add_effects == {"inbox", "handempty"}
    anonymous = build_partial_domain(_OPTIONS, _PREDICATES, hints,
                                     NameMap(_OPTIONS, _PREDICATES, False))
    assert "placetoy" not in anonymous and "inbox" not in anonymous
    assert "?toy0 - toy" in anonymous


@pytest.mark.parametrize("entry, message", [
    ({"name": "Holding"}, "not a given predicate"),
    ({"name": "inbox"}, "not a given predicate"),
    ({"name": "InBox", "types": ["toy"]}, "unknown keys"),
    ({"name": "InBox", "effects": {"Wipe": "add"}}, "unknown option"),
    ({"name": "InBox", "effects": {"PickToy": "maybe"}}, "must be"),
])
def test_bad_known_predicates(entry, message):
    """Names and effects that do not match the environment are errors."""
    with pytest.raises(ValueError, match=message):
        _hints([entry])


_REPLY = """Here is the domain:
```pddl
(define (domain robot)
  ; comment (with parens
  (:predicates (inbox ?t - toy) (handempty ?r - robot)
               (holding ?r - robot ?t - toy) (seen ?t - toy))
  (:action picktoy :parameters (?r - robot ?t - toy)
    :precondition (and (handempty ?r))
    :effect (and (holding ?r ?t) (not (handempty ?r))))
  (:action PlaceToy :parameters (?r - robot ?t - toy)
    :effect (and (not (holding ?r ?t)) (inbox ?t)))
  (:action unknownaction :parameters () :effect (seen ?t)))
```"""


def test_extract_proposals():
    """New predicates with effects become effect vectors; known ones, unknown
    actions and predicates without effects are ignored."""
    names = NameMap(_OPTIONS, _PREDICATES, semantic=True)
    proposals = extract_proposals(parse_domain(_REPLY), _OPTIONS, names)
    assert len(proposals) == 1
    assert proposals[0].name == "holding"
    assert proposals[0].types == ("robot", "toy")
    assert proposals[0].codes == (ADD, DELETE)
    assert parse_domain("no domain here") is None


def test_complete_domain_with_canned_reply(tmp_path):
    """complete_domain keeps the proposals with an allowed signature and
    groups them by target."""
    responses = tmp_path / "responses.json"
    responses.write_text(
        json.dumps([{
            "task": "pddl_completion",
            "response": _REPLY
        }]))
    client = LLMClient("mock", "mock", 0.2, 100, 10, 2,
                       mock_responses=str(responses))
    names = NameMap(_OPTIONS, _PREDICATES, semantic=True)
    proposals = complete_domain(client, "(define (domain robot))", "", "",
                                {("robot", "toy")}, _OPTIONS, names)
    assert [p.name for p in proposals] == ["holding"]
    target = Predicate("neural_b_robot_toy", [_ROBOT, _TOY],
                       lambda s, o: True)
    other = Predicate("neural_u_toy", [_TOY], lambda s, o: True)
    seeds = proposals_to_seeds(proposals, [target, other])
    assert [list(s) for s in seeds[target]] == [[ADD, DELETE]]
    assert seeds[other] == []
    assert complete_domain(client, "", "", "", {("toy", )}, _OPTIONS,
                           names) == []
