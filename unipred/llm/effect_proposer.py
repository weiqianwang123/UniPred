"""LLM-in-the-loop effect proposals (stage 3 of UniPred).

An EffectProposer proposes effect vectors for one predicate type: first the
seeds from PDDL completion, then (if the LLM loop is on) new proposals from
the LLM, which sees every earlier proposal with a score derived from the
validation loss of the classifier trained on it.

It implements the searcher interface that
BilevelLearningApproach.gen_sat_vec() expects (propose() and
update_value()), so IVNTR's constraint check, classifier training and
acceptance test are used unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.structs import ParameterizedOption, Predicate
from unipred.llm.client import LLMClient
from unipred.llm.names import NameMap
from unipred.llm.pddl import ADD, DELETE, NO_EFFECT, load_prompt

Codes = Tuple[int, ...]


def loss_to_score(loss: Optional[float], accept_loss: float,
                  zero_loss: float) -> int:
    """Map a validation loss to a score in [0, 100].

    The mapping is monotonic and linear in log(loss): a loss at or below the
    acceptance threshold scores 100 and a loss at or above zero_loss scores
    0. Losses span orders of magnitude, so a log scale keeps differences
    between good proposals visible; unlike normalizing across the
    proposals, a score does not change when new proposals arrive.
    """
    assert 0 < accept_loss < zero_loss
    if loss is None or not math.isfinite(loss):
        return 0
    if loss <= accept_loss:
        return 100
    fraction = math.log(zero_loss / max(loss, 1e-12)) / math.log(
        zero_loss / accept_loss)
    return int(round(100 * min(max(fraction, 0.0), 1.0)))


def allowed_codes(num_rows: int,
                  constraints: Sequence[Tuple]) -> List[Set[int]]:
    """The effect codes that IVNTR's position constraints allow per row."""
    allowed = [{NO_EFFECT, ADD, DELETE} for _ in range(num_rows)]
    for rule in constraints:
        if rule[0] != "position":
            continue
        row, _, channel, value = rule[1:]
        code = ADD if channel == 0 else DELETE
        if value:
            allowed[row] = {code}
        else:
            allowed[row].discard(code)
    return allowed


class EffectProposer:
    """Proposes effect vectors of one predicate type (see module doc)."""

    def __init__(self, client: Optional[LLMClient], target: Predicate,
                 rows: Sequence[ParameterizedOption],
                 known_predicates: Sequence[Predicate], names: NameMap,
                 allowed: Sequence[Set[int]], seeds: Sequence[np.ndarray],
                 use_llm: bool, loss_feedback: bool, description: str,
                 accept_loss: float, zero_loss: float) -> None:
        assert client is not None or not use_llm
        self._client = client
        self._target = target
        self._rows = list(rows)
        self._names = names
        self._allowed = [set(a) for a in allowed]
        self._seeds = [tuple(int(c) for c in s) for s in seeds]
        self._use_llm = use_llm
        self._loss_feedback = loss_feedback
        self._accept_loss = accept_loss
        self._zero_loss = zero_loss
        # Every proposal in order, with its validation loss (None until it
        # is trained, inf if it violates the constraints).
        self._history: Dict[Codes, Optional[float]] = {}
        self._num_requests = 0
        self._system = self._system_prompt(known_predicates, description)

    @property
    def history(self) -> Dict[Codes, Optional[float]]:
        """The proposals so far and their validation losses."""
        return dict(self._history)

    def propose(self) -> Optional[np.ndarray]:
        """The next effect vector to train, or None if there is none."""
        for seed in self._seeds:
            if seed not in self._history:
                self._history[seed] = None
                logging.info(f"Proposing PDDL seed for {self._target.name}: "
                             f"{self._describe(seed)}")
                return np.array(seed)
        if not self._use_llm:
            return None
        assert self._client is not None
        notes: List[str] = []
        for attempt in range(self._client.retry_attempts):
            reply = self._client.complete(self._system,
                                          self._user_prompt(notes),
                                          "effect_proposal",
                                          self._num_requests * 100 + attempt)
            codes, problem = self._parse(reply)
            if codes is not None:
                self._num_requests += 1
                self._history[codes] = None
                logging.info(f"LLM proposes for {self._target.name}: "
                             f"{self._describe(codes)}")
                return np.array(codes)
            logging.info(f"Rejected LLM reply for {self._target.name}: "
                         f"{problem}")
            notes.append(problem)
        self._num_requests += 1
        logging.info(f"The LLM gave no valid new proposal for "
                     f"{self._target.name}.")
        return None

    def update_value(self, state: np.ndarray, guidance: np.ndarray) -> None:
        """Called by gen_sat_vec() when a proposal violates the
        constraints."""
        del guidance  # unused
        self._history[tuple(int(c) for c in state)] = float("inf")

    def record_losses(self, vectors: Sequence[np.ndarray],
                      losses: Sequence[float]) -> None:
        """Record the validation losses of trained proposals."""
        for vector, loss in zip(vectors, losses):
            self._history[tuple(int(c) for c in vector)] = float(loss)

    def _describe(self, codes: Codes) -> str:
        adds = [self._names.option(o) for o, c in zip(self._rows, codes)
                if c == ADD]
        dels = [self._names.option(o) for o, c in zip(self._rows, codes)
                if c == DELETE]
        return f"ADD: {', '.join(adds)} | DEL: {', '.join(dels)}"

    def _system_prompt(self, known_predicates: Sequence[Predicate],
                       description: str) -> str:
        predicates = [
            f"  {self._names.predicate(p)}({', '.join(t.name for t in p.types)})"
            for p in sorted(known_predicates, key=self._names.predicate)
        ]
        actions = [
            f"  {self._names.option(o)}({', '.join(t.name for t in o.types)})"
            for o in self._rows
        ]
        add_candidates = [
            self._names.option(o) for o, a in zip(self._rows, self._allowed)
            if ADD in a
        ]
        del_candidates = [
            self._names.option(o) for o, a in zip(self._rows, self._allowed)
            if DELETE in a
        ]
        return load_prompt("effect_proposal_system.txt").substitute(
            description=description or "(none)",
            target=self._target.name,
            target_types=", ".join(t.name for t in self._target.types),
            predicates="\n".join(predicates) or "  (none)",
            actions="\n".join(actions),
            add_candidates=", ".join(add_candidates) or "(none)",
            del_candidates=", ".join(del_candidates) or "(none)",
            feedback=("Each tried effect pattern has a score from 0 to 100 "
                      "that says how well a classifier with these effects "
                      "fits the robot's data; 100 means it fits well enough "
                      "to be kept." if self._loss_feedback else
                      "You see the effect patterns tried so far."))

    def _user_prompt(self, notes: Sequence[str]) -> str:
        if not self._history:
            lines = ["No effect patterns have been tried yet."]
        else:
            lines = ["Effect patterns tried so far:"]
            entries = list(self._history.items())
            if self._loss_feedback:
                entries.sort(key=lambda e: -self._score(e[1]))
            for codes, loss in entries:
                line = f"- {self._describe(codes)}"
                if self._loss_feedback:
                    if loss is None:
                        line += " -> not scored yet"
                    elif math.isinf(loss):
                        line += " -> impossible for these actions"
                    else:
                        line += f" -> score {self._score(loss)}"
                lines.append(line)
        for note in notes:
            lines.append(f"Your previous reply was invalid: {note}")
        lines.append("Propose a new effect pattern that is not in the list.")
        return "\n".join(lines)

    def _score(self, loss: Optional[float]) -> int:
        if loss is None:
            return -1
        return loss_to_score(loss, self._accept_loss, self._zero_loss)

    def _parse(self, reply: str) -> Tuple[Optional[Codes], str]:
        fields: Dict[str, str] = {}
        for line in reply.splitlines():
            line = line.strip().strip("*").strip()
            for key in ("ADD", "DEL"):
                if line.upper().startswith(key + ":"):
                    fields[key] = line.split(":", 1)[1]
        if set(fields) != {"ADD", "DEL"}:
            return None, "it must contain an ADD: line and a DEL: line."
        codes = [NO_EFFECT] * len(self._rows)
        row_of = {o: i for i, o in enumerate(self._rows)}
        for key, code in (("ADD", ADD), ("DEL", DELETE)):
            for name in fields[key].split(","):
                name = name.strip().strip(".").strip()
                if not name or name.lower() in ("none", "-", "blank"):
                    continue
                try:
                    row = row_of[self._names.option_from_symbol(name)]
                except KeyError:
                    return None, f"{name} is not an action."
                if codes[row] != NO_EFFECT:
                    return None, f"{name} is listed twice."
                if code not in self._allowed[row]:
                    kind = "ADD" if code == ADD else "DEL"
                    return None, f"{name} is not an {kind} candidate."
                codes[row] = code
        # Rows whose effect the data determines are filled in.
        for row, allowed in enumerate(self._allowed):
            if codes[row] == NO_EFFECT and NO_EFFECT not in allowed:
                codes[row] = min(allowed)
        if not any(codes):
            return None, "the pattern has no effects."
        if tuple(codes) in self._history:
            return None, (f"'{self._describe(tuple(codes))}' was already "
                          "tried.")
        return tuple(codes), ""
