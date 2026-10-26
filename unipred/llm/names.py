"""Names of options, predicates and types as the LLM sees them.

With semantic names, a name is only normalized into a PDDL symbol. Without
them (the "w/o Semantic" ablation), options and known predicates are
replaced by neutral ids, also where the domain description mentions them;
everything else, including the type names, stays visible, as in the paper.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set

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
        # Original name -> shown name, for the text the LLM reads.
        self._renames = {
            item.name: shown
            for item, shown in [*self._option_names.items(),
                                *self._predicate_names.items()]
            if pddl_symbol(item.name) != shown
        }

    def option(self, option: ParameterizedOption) -> str:
        """The name of an option."""
        return self._option_names[option]

    def predicate(self, pred: Predicate) -> str:
        """The name of a known predicate."""
        return self._predicate_names[pred]

    def known_predicate_symbols(self) -> Set[str]:
        """The names of all known predicates."""
        return set(self._predicate_names.values())

    def option_from_symbol(self, symbol: str) -> ParameterizedOption:
        """The option with the given name; raises KeyError if none."""
        return self._symbol_to_option[pddl_symbol(symbol)]

    def text(self, text: str) -> str:
        """The text with every hidden option or predicate name (a whole
        word, in any case) replaced by its neutral id."""
        for name, shown in sorted(self._renames.items(),
                                  key=lambda item: -len(item[0])):
            text = re.sub(rf"\b{re.escape(name)}\b",
                          shown,
                          text,
                          flags=re.IGNORECASE)
        return text
