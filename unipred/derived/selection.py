"""Derived-aware predicate selection (Algorithm 1 of the paper).

IVNTR adds negated and quantified variants of every learned binary
predicate to the selection candidates. Each candidate carries a tag
(template, group, variant): variant 0 is the learned basic predicate and
variants 1, 2, ... are its derived forms. Plain IVNTR treats the variants
like any other predicate, so they must be consistent operator effects, which
derived predicates often are not (e.g. "no toy is on the table" is a
precondition of WipeTable but no operator's effect).

With derived-aware selection:

- the variants are left out of the initial candidate pool; once their basic
  predicate is selected, they are added to the pool;
- operators are learned from the basic predicates only; the variants can
  appear in preconditions but never in effects;
- while scoring a predicate set and while planning, an atoms closure (see
  unipred/derived/closure.py) recomputes the variants from the basic atoms.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from predicators import hooks, utils
from predicators.approaches.bilevel_learning_approach import \
    PredicateCandidate
from predicators.gnn.neupi_utils import parse_basic_pred_name
from predicators.hooks import AtomsClosure
from predicators.structs import Dataset, Predicate
from unipred.derived.closure import make_closure
from unipred.derived.rules import DerivedRule

class DerivedSelectionMixin:
    """Derived-aware selection for subclasses of BilevelLearningApproach.

    It must come before BilevelLearningApproach in the bases.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        # Rules of the derived candidates, by head predicate.
        self._derived_rules: Dict[Predicate, DerivedRule] = {}
        # Derived candidates that enter the pool with their basic predicate.
        self._withheld_variants: Dict[Predicate,
                                      List[PredicateCandidate]] = {}

    def _use_derived_selection(self) -> bool:
        """Whether derived-aware selection is on."""
        return True

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Check the planner settings, learn, and save the derived rules."""
        if self._use_derived_selection():
            hooks.check_closure_settings()
        super().learn_from_offline_dataset(dataset)  # type: ignore[misc]
        learned = self._learned_predicates  # type: ignore[attr-defined]
        specs = {
            rule.head.name: {
                "base": rule.base.name,
                "negated": rule.negated,
                "quantifier": rule.quantifier,
                "quantified": list(rule.quantified),
            }
            for rule in self._derived_rules.values() if rule.head in learned
        }
        with open(f"{utils.get_approach_save_path_str()}.derived",
                  "w",
                  encoding="utf-8") as f:
            json.dump(specs, f, indent=1)

    def load(self) -> None:
        """Load the approach and its derived rules."""
        super().load()  # type: ignore[misc]
        self._derived_rules = {}
        self._withheld_variants = {}
        path = f"{utils.get_approach_load_path_str()}.derived"
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            specs = json.load(f)
        by_name = {
            p.name: p
            for p in self._learned_predicates  # type: ignore[attr-defined]
        }
        for head_name, spec in specs.items():
            self._derived_rules[by_name[head_name]] = DerivedRule(
                head=by_name[head_name],
                base=by_name[spec["base"]],
                negated=spec["negated"],
                quantifier=spec["quantifier"],
                quantified=tuple(spec["quantified"]))

    def _select_predicates_by_score_search(
        self, huge_ae_matrixe: torch.Tensor,
        all_predicates_info: Dict[int, Tuple[Predicate, List[int], bool,
                                             Tuple]], score_function
    ) -> Tuple[Set[Predicate], torch.Tensor, List[Predicate], List[List[int]]]:
        if not self._use_derived_selection():
            return super()._select_predicates_by_score_search(  # type: ignore[misc]
                huge_ae_matrixe, all_predicates_info, score_function)
        groups: Dict[Tuple[Predicate, int], Dict[int, int]] = {}
        for col, (_, _, provided, tag) in all_predicates_info.items():
            if not provided:
                template, group, variant = tag
                groups.setdefault((template, group), {})[variant] = col
        variant_cols = set()
        for (template, group), variants in groups.items():
            base_col = variants.get(0)
            for variant, col in variants.items():
                if variant == 0:
                    continue
                variant_cols.add(col)
                head, ent_idx = all_predicates_info[col][:2]
                if base_col is None:
                    logging.info(f"Dropping {head.name}: its basic predicate "
                                 "was pruned as a duplicate.")
                    continue
                base = all_predicates_info[base_col][0]
                rule = self._rule(template, group, variant, head, base)
                self._derived_rules[head] = rule
                self._withheld_variants.setdefault(base, []).append(
                    PredicateCandidate(head, huge_ae_matrixe[:, col],
                                       ent_idx))
                logging.info(f"Derived candidate {head.name} of "
                             f"{base.name}: {rule}")
        basic_info = {
            col: info
            for col, info in all_predicates_info.items()
            if col not in variant_cols
        }
        return super()._select_predicates_by_score_search(  # type: ignore[misc]
            huge_ae_matrixe, basic_info, score_function)

    def _rule(self, template: Predicate, group: int, variant: int,
              head: Predicate, base: Predicate) -> DerivedRule:
        info = self.learned_ae_pred_info[template]  # type: ignore[attr-defined]
        quantifier_str, neg_str, basic_pred_str = \
            info["quantifiers"][group][variant]
        quantified, _ = parse_basic_pred_name(
            basic_pred_str, [t.name for t in template.types])
        quantifier = quantifier_str.split(":")[0] if "?" in quantifier_str \
            else None
        return DerivedRule(head=head,
                           base=base,
                           negated=bool(neg_str),
                           quantifier=quantifier,
                           quantified=tuple(quantified))

    def _expand_candidate_pool(self, selected: List[Predicate],
                               pool: List[PredicateCandidate]) -> None:
        in_pool = {candidate.predicate for candidate in pool}
        for candidate in self._withheld_variants.get(selected[-1], []):
            if candidate.predicate not in in_pool:
                logging.info(f"Adding derived candidate "
                             f"{candidate.predicate.name} to the pool.")
                pool.append(candidate)

    def _operator_predicates(
            self, predicates: List[Predicate]) -> List[Predicate]:
        return [p for p in predicates if p not in self._derived_rules]

    def _atoms_closure(
            self, predicates: Sequence[Predicate]) -> Optional[AtomsClosure]:
        rules = [self._derived_rules[p] for p in predicates
                 if p in self._derived_rules]
        return make_closure(rules) if rules else None
