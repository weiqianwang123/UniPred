"""Offline answers for the "mock" LLM provider (tests and smoke runs).

A JSON file given by ``--llm_mock_responses`` can provide canned replies::

    [{"task": "pddl_completion", "contains": "satellite", "response": "..."}]

The first entry whose task matches and whose "contains" text occurs in the
prompt is returned. Otherwise, simple rules give valid, deterministic
answers: PDDL completion invents one predicate per allowed type list, and
effect proposals walk through the ADD and DEL candidates.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from unipred.llm.pddl import parse_domain


def mock_response(system: str, user: str, task: str,
                  responses_path: str) -> str:
    """The mock reply to a request."""
    if responses_path:
        with open(responses_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                if entry["task"] == task and entry.get("contains",
                                                       "") in system + user:
                    return entry["response"]
    if task == "pddl_completion":
        return _complete_pddl(user)
    if task == "effect_proposal":
        return _propose_effects(user)
    raise ValueError(f"The mock LLM cannot answer task {task!r}.")


def _signatures(user: str) -> List[Tuple[str, ...]]:
    section = user.split("New predicates may only take these argument types",
                         1)[-1]
    return [
        tuple(t.strip() for t in match.split(","))
        for match in re.findall(r"^- \(([^)]*)\)", section, re.MULTILINE)
    ]


def _complete_pddl(user: str) -> str:
    domain = parse_domain(user)
    assert domain is not None, "The prompt has no PDDL domain."
    declarations = []
    actions = {name: ([], []) for name in domain.actions}
    for k, signature in enumerate(_signatures(user)):
        name = f"mockpred{k}"
        args = " ".join(f"?x{i} - {t}" for i, t in enumerate(signature))
        declarations.append(f"({name} {args})")
        fitting = [
            action_name
            for action_name, action in sorted(domain.actions.items())
            if all(
                sum(t == s for _, t in action.parameters) >= signature.count(s)
                for s in signature)
        ]
        if not fitting:
            continue
        actions[fitting[0]][0].append(name)
        if len(fitting) > 1:
            actions[fitting[-1]][1].append(name)
    blocks = []
    for action_name, (adds, dels) in actions.items():
        effects = [f"({p})" for p in adds] + [f"(not ({p}))" for p in dels]
        blocks.append(f"(:action {action_name} :parameters () "
                      f":effect (and {' '.join(effects)}))")
    return (f"(define (domain mock) (:predicates {' '.join(declarations)}) "
            f"{' '.join(blocks)})")


def _candidates(user: str, key: str) -> List[str]:
    match = re.search(rf"^Actions that may {key} it: (.*)$", user,
                      re.MULTILINE)
    if match is None or match.group(1).strip() == "(none)":
        return []
    return [name.strip() for name in match.group(1).split(",")]


def _propose_effects(user: str) -> str:
    adds = _candidates(user, "ADD")
    dels = _candidates(user, "DEL")
    patterns = [([a], []) for a in adds] + [([], [d]) for d in dels] + [
        ([a], [d]) for a in adds for d in dels if a != d
    ]
    assert patterns, "No ADD or DEL candidates in the prompt."
    num_seen = len(re.findall(r"^- ADD:", user, re.MULTILINE)) + len(
        re.findall(r"^Your previous reply was invalid", user, re.MULTILINE))
    add, delete = patterns[num_seen % len(patterns)]
    return (f"ADD: {', '.join(add)}\nDEL: {', '.join(delete)}\n"
            "WHY: mock proposal")
