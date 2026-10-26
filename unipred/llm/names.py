"""Names of options, predicates and types as the LLM sees them.

With semantic names, a name is only normalized into a PDDL symbol. Without
them (the "w/o Semantic" ablation), options and known predicates are
replaced by neutral ids; type names stay visible, as in the paper.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable

from predicators.structs import ParameterizedOption, Predicate


def pddl_symbol(name: str) -> str:
    """Normalize a name into a lowercase PDDL symbol."""
    return re.sub(r"[^a-z0-9_]+", "_", name.lower())


class NameMap:
    """Maps options and known predicates to the names shown to the LLM, and
    LLM symbols back to options."""

    def __init__(self, options: Iterable[ParameterizedOption],
                 predicates: Iterable[Predicate], semantic: bool) -> None:
        self._option_names: Dict[ParameterizedOption, str] = {}
        self._predicate_names: Dict[Predicate, str] = {}
        for i, option in enumerate(sorted(options, key=lambda o: o.name)):
            self._option_names[option] = pddl_symbol(
                option.name) if semantic else f"action{i}"
        for i, pred in enumerate(sorted(predicates, key=lambda p: p.name)):
            self._predicate_names[pred] = pddl_symbol(
                pred.name) if semantic else f"pred{i}"
        self._symbol_to_option = {
            symbol: option
            for option, symbol in self._option_names.items()
        }
        assert len(self._symbol_to_option) == len(self._option_names), \
            "Option names collide after normalization."

    def option(self, option: ParameterizedOption) -> str:
        """The name of an option."""
        return self._option_names[option]

    def predicate(self, pred: Predicate) -> str:
        """The name of a known predicate."""
        return self._predicate_names[pred]

    def known_predicate_symbols(self) -> set:
        """The names of all known predicates."""
        return set(self._predicate_names.values())

    def option_from_symbol(self, symbol: str) -> ParameterizedOption:
        """The option with the given name; raises KeyError if none."""
        return self._symbol_to_option[pddl_symbol(symbol)]
