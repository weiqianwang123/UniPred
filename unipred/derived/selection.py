"""Derived-aware predicate selection (Algorithm 1 of the paper).

IVNTR adds negated and quantified variants of every learned binary
predicate to the selection candidates. Each candidate carries a tag
(template, group, variant): variant 0 is the learned basic predicate and
variants 1, 2, ... are its derived forms. Plain IVNTR treats the variants
like any other predicate, so they must be consistent operator effects, which
derived predicates often are not (e.g. "no toy is on the table" is a
precondition of WipeTable but no operator's effect).

With derived-aware selection:

- a variant is a candidate if it passes IVNTR's check (consistent effects,
  and a precondition where it is no effect) or else is a precondition of
  some option (holding before all its transitions that leave it unchanged,
  without holding everywhere): IVNTR keeps only the first kind, which
  drops "no toy is on the table";
- the variants are left out of the initial candidate pool; once their basic
  predicate is selected, they are added to the pool; and a basic candidate
  is also scored together with each of its variants (and added with the
  best one), as a basic predicate may help only through a derived form;
- operators are learned from the basic predicates only; the variants can
  appear in preconditions but never in effects;
- while scoring a predicate set and while planning, an atoms closure (see
  unipred/derived/closure.py) recomputes the variants from the basic atoms;
- selection does not end as soon as the score is below the threshold: it
  goes on adding candidates that lower the score by at least
  REFINE_MIN_IMPROVEMENT, since derived predicates often only tighten
  preconditions (wiping before the toys are away is ruled out only by "no
  toy is on the table"), which the demonstrations' score barely needs;
- after the operators are learned, each derived precondition is dropped if
  the planning effort on the demonstrations stays within PRUNE_TOLERANCE
  of the operators' with all of them: demonstrations in one order make "no
  toy is on the table" hold before every later operator (picking the
  sponge, pushing the tray), and those preconditions would make tasks that
  start in another order (the sponge already in hand) unsolvable.
"""

from __future__ import annotations

import contextlib
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
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.predicate_search_score_functions import \
    _OperatorBeliefScoreFunction
from predicators.settings import CFG
from predicators.structs import NSRT, Dataset, GroundAtomTrajectory, \
    LowLevelTrajectory, Predicate
from unipred.derived.closure import make_closure
from unipred.derived.rules import DerivedRule

# Below the selection threshold, a candidate must lower the score by this
# fraction to be added.
REFINE_MIN_IMPROVEMENT = 0.01
# A derived precondition is dropped if the score rises by at most this
# fraction without it.
PRUNE_TOLERANCE = 0.01

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

    def _derived_variants_need_effects(self) -> bool:
        """Variants only enter preconditions (see module doc)."""
        return not self._use_derived_selection()

    def _candidate_bundles(
            self,
            candidate: PredicateCandidate) -> List[List[PredicateCandidate]]:
        """A basic candidate alone and with each of its variants (see module
        doc)."""
        bundles = [[candidate]]
        if self._use_derived_selection():
            bundles += [[candidate, variant] for variant in
                        self._withheld_variants.get(candidate.predicate, [])]
        return bundles

    def _refine_below_threshold(self) -> Optional[float]:
        """Selection goes on below the threshold (see module doc)."""
        return REFINE_MIN_IMPROVEMENT if self._use_derived_selection() \
            else None

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

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     atom_dataset: Optional[List[GroundAtomTrajectory]],
                     annotations: Optional[List]) -> None:
        super()._learn_nsrts(  # type: ignore[misc]
            trajectories, atom_dataset, annotations)
        if self._use_derived_selection() and atom_dataset is not None:
            self._prune_derived_preconditions(atom_dataset)

    def _prune_derived_preconditions(
            self, atom_dataset: List[GroundAtomTrajectory]) -> None:
        """Drop the derived preconditions that the planning effort on the
        demonstrations does not need (see module doc)."""
        learned = sorted(self._learned_predicates)  # type: ignore[attr-defined]
        predicates = set(learned) | set(
            self._initial_predicates)  # type: ignore[attr-defined]
        trajectories = [traj for traj, _ in atom_dataset]
        segmented = [
            segment_trajectory(traj, predicates, atoms)
            for traj, atoms in atom_dataset
        ]
        score_function = _OperatorBeliefScoreFunction(
            atom_dataset,
            self._train_tasks,  # type: ignore[attr-defined]
            self.ae_row_names,  # type: ignore[attr-defined]
            CFG.neupi_selection_metric)
        closure = self._atoms_closure(learned)

        def _score(nsrts: List[NSRT]) -> float:
            context = hooks.temporary_atoms_closure(closure) \
                if closure is not None else contextlib.nullcontext()
            with context:
                return score_function.evaluate_with_operators(
                    predicates, trajectories, segmented,
                    [n.op for n in nsrts],
                    [(n.option, list(n.option_vars)) for n in nsrts])

        nsrts = sorted(self._nsrts)  # type: ignore[attr-defined]
        limit = _score(nsrts) * (1 + PRUNE_TOLERANCE)
        for i in range(len(nsrts)):
            for atom in sorted(nsrts[i].preconditions, key=str):
                if atom.predicate not in self._derived_rules:
                    continue
                trial = list(nsrts)
                trial[i] = nsrts[i].op.copy_with(
                    preconditions=nsrts[i].preconditions - {atom}).make_nsrt(
                        nsrts[i].option, nsrts[i].option_vars,
                        nsrts[i].sampler)
                score = _score(trial)
                verdict = "dropped" if score <= limit else "kept"
                logging.info(f"Derived precondition {atom} of "
                             f"{nsrts[i].name}: score without it {score} "
                             f"(limit {limit}), {verdict}.")
                if score <= limit:
                    nsrts = trial
        self._nsrts = set(nsrts)
        logging.info("NSRTs after pruning derived preconditions:")
        for nsrt in nsrts:
            logging.info(nsrt)

    def _operator_predicates(
            self, predicates: List[Predicate]) -> List[Predicate]:
        return [p for p in predicates if p not in self._derived_rules]

    def _atoms_closure(
            self, predicates: Sequence[Predicate]) -> Optional[AtomsClosure]:
        rules = [self._derived_rules[p] for p in predicates
                 if p in self._derived_rules]
        return make_closure(rules) if rules else None
