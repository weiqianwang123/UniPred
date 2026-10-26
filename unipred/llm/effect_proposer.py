"""LLM-in-the-loop effect proposals (stage 3 of UniPred).

An EffectProposer proposes effect vectors for one predicate type: first the
seeds from PDDL completion, then (if it has an LLM client, i.e. the LLM
loop is on) new proposals from the LLM. The LLM sees the domain, the demonstrations and every earlier
proposal with the feedback from the classifier trained on it: a score in
[0, 100] derived from the validation loss, or, without loss feedback,
only whether the proposal was accepted.

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
from unipred.llm.pddl import ADD, DELETE, NO_EFFECT, PromptContext, \
    load_prompt

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
                 rows: Sequence[ParameterizedOption], names: NameMap,
                 context: PromptContext, allowed: Sequence[Set[int]],
                 seeds: Sequence[np.ndarray], loss_feedback: bool,
                 accept_loss: float, zero_loss: float) -> None:
        self._client = client
        self._target = target
        self._rows = list(rows)
        self._names = names
        self._context = context
        self._allowed = [set(a) for a in allowed]
        self._seeds = [tuple(int(c) for c in s) for s in seeds]
        self._loss_feedback = loss_feedback
        self._accept_loss = accept_loss
        self._zero_loss = zero_loss
        # Every proposal in order, with its validation loss (None until it
        # is trained, inf if it violates the constraints).
        self._history: Dict[Codes, Optional[float]] = {}
        self._num_requests = 0

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
        if self._client is None:
            return None
        system = load_prompt("effect_proposal_system.txt").substitute()
        notes: List[str] = []
        for attempt in range(self._client.retry_attempts):
            reply = self._client.complete(system, self._user_prompt(notes),
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
        adds = [
            self._names.option(o) for o, c in zip(self._rows, codes)
            if c == ADD
        ]
        dels = [
            self._names.option(o) for o, c in zip(self._rows, codes)
            if c == DELETE
        ]
        return f"ADD: {', '.join(adds)} | DEL: {', '.join(dels)}"

    def _user_prompt(self, notes: Sequence[str]) -> str:
        add_candidates = [
            self._names.option(o) for o, a in zip(self._rows, self._allowed)
            if ADD in a
        ]
        del_candidates = [
            self._names.option(o) for o, a in zip(self._rows, self._allowed)
            if DELETE in a
        ]
        history = [
            f"- {self._describe(codes)} -> {self._feedback(loss)}"
            for codes, loss in self._history.items()
        ] or ["(none)"]
        history += [f"Your previous reply was invalid: {n}" for n in notes]
        if self._loss_feedback:
            feedback = ("A score from 0 to 100 says how well a classifier "
                        "with the effect pattern fits the robot's data; 100 "
                        "means it fits well enough to be kept.")
        else:
            feedback = ("'accepted' means a classifier with the effect "
                        "pattern fits the robot's data well enough to be "
                        "kept.")
        return load_prompt("effect_proposal_user.txt").substitute(
            description=self._context.description or "(none)",
            domain=self._context.domain,
            demos=self._context.demos or "(none)",
            target=self._target.name,
            target_types=", ".join(t.name for t in self._target.types),
            add_candidates=", ".join(add_candidates) or "(none)",
            del_candidates=", ".join(del_candidates) or "(none)",
            history="\n".join(history),
            feedback=feedback)

    def _feedback(self, loss: Optional[float]) -> str:
        """What the LLM learns about a proposal."""
        if loss is None:
            return "not trained yet"
        if math.isinf(loss):
            return "impossible for these actions"
        if self._loss_feedback:
            return "score " + str(
                loss_to_score(loss, self._accept_loss, self._zero_loss))
        return "accepted" if loss <= self._accept_loss else "rejected"

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
                    return None, f"{name} cannot {kind} this predicate."
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
