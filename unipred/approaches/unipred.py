"""UniPred: deep predicate invention guided by an LLM.

UniPred is IVNTR's bilevel learning (predicators/approaches/
bilevel_learning_approach.py) with the effect vectors of each predicate
type proposed by an LLM instead of a tree search:

- stage 2, PDDL completion (unipred/llm/pddl.py): the LLM completes a
  partial PDDL domain; its new predicates seed the search;
- stage 3, LLM-in-the-loop learning (unipred/llm/effect_proposer.py): the
  LLM proposes further effect vectors given the scores of earlier ones;
  IVNTR trains a classifier for each and keeps those that fit the data;
- stage 4, derived-aware selection (unipred/derived/selection.py);
- stage 5, operator and sampler learning, as in IVNTR.

Every stage is a setting (see unipred/settings.py); turning one off gives
the corresponding ablation of the paper.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.bilevel_learning_approach import \
    BilevelLearningApproach, read_predicate_config
from predicators.gnn.neupi_utils import two2one
from predicators.settings import CFG
from predicators.structs import Dataset, DummyPredicate, \
    ParameterizedOption, Predicate, Task, Type
from unipred.derived.selection import DerivedSelectionMixin
from unipred.llm.client import LLMClient
from unipred.llm.effect_proposer import EffectProposer, allowed_codes
from unipred.llm.names import NameMap, pddl_symbol
from unipred.llm.pddl import build_partial_domain, complete_domain, \
    load_known_predicates, proposals_to_seeds, render_demos


class UniPredApproach(DerivedSelectionMixin, BilevelLearningApproach):
    """IVNTR with LLM-proposed effect vectors (see module doc)."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        if CFG.neupi_gt_ae_matrix:
            raise ValueError("UniPred proposes effect vectors itself; set "
                             "--neupi_gt_ae_matrix False.")
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # The description and known predicates of the domain.
        domain_config = read_predicate_config(CFG.neupi_pred_config)
        self._domain_description: str = domain_config.get("description", "")
        self._known_predicate_entries: List[Dict] = domain_config.get(
            "known_predicates", [])
        self._client: Optional[LLMClient] = None
        self._proposers: Dict[Predicate, EffectProposer] = {}
        self._seeds: Optional[Dict[Predicate, List[np.ndarray]]] = None
        self._dataset: Optional[Dataset] = None

    @classmethod
    def get_name(cls) -> str:
        return "unipred"

    def _use_derived_selection(self) -> bool:
        return CFG.unipred_derived_selection

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._dataset = dataset
        super().learn_from_offline_dataset(dataset)
        if self._client is not None:
            logging.info(f"LLM usage: {self._client.usage}")
        self._save_llm_history()

    def gen_ae_vectors4pred(self, iteration, curr_pred, ent_idx,
                            pred_save_path, pred_config, data,
                            symbolic_model):
        """Propose effect vectors with the EffectProposer of the predicate
        instead of the tree search, then report their validation losses."""
        del symbolic_model  # replaced by the proposer
        proposer = self._proposer(curr_pred, pred_config)
        ae_vecs, scores, val_losses, model_paths = \
            super().gen_ae_vectors4pred(iteration, curr_pred, ent_idx,
                                        pred_save_path, pred_config, data,
                                        proposer)
        proposer.record_losses([two2one(v).numpy() for v in ae_vecs],
                               val_losses)
        return ae_vecs, scores, val_losses, model_paths

    def _llm(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient.from_settings()
        return self._client

    def _names(self) -> NameMap:
        return NameMap(self.ae_row_names, self._initial_predicates,
                       CFG.unipred_semantic_names)

    def _description(self) -> str:
        # The description would reveal the names in the w/o Semantic
        # ablation.
        if not CFG.unipred_semantic_names:
            return ""
        return self._domain_description

    def _targets(self) -> List[DummyPredicate]:
        return [
            pred for pred, info in self.learned_ae_pred_info.items()
            if not info["provided"]
        ]

    def _seed_vectors(self) -> Dict[Predicate, List[np.ndarray]]:
        """Effect vectors of the predicates proposed by PDDL completion,
        by target predicate type (computed once)."""
        if self._seeds is not None:
            return self._seeds
        targets = self._targets()
        self._seeds = {target: [] for target in targets}
        if not CFG.unipred_pddl_completion:
            return self._seeds
        assert self._dataset is not None
        names = self._names()
        hints = load_known_predicates(self._known_predicate_entries,
                                      self._initial_predicates,
                                      self.ae_row_names)
        domain = build_partial_domain(self.ae_row_names,
                                      self._initial_predicates, hints, names)
        demos = render_demos(self._dataset.trajectories, self._train_tasks,
                             self._initial_predicates, names,
                             CFG.unipred_num_demo_prompts)
        signatures = {
            tuple(pddl_symbol(t.name) for t in target.types)
            for target in targets
        }
        proposals = complete_domain(self._llm(), domain, demos,
                                    self._description(), signatures,
                                    self.ae_row_names, names)
        self._seeds = proposals_to_seeds(proposals, targets)
        return self._seeds

    def _proposer(self, curr_pred: DummyPredicate,
                  pred_config: Dict) -> EffectProposer:
        if curr_pred not in self._proposers:
            allowed = allowed_codes(
                len(self.ae_row_names),
                self.learned_ae_pred_info[curr_pred]["constraints"])
            # A predicate type can train only its PDDL seeds.
            use_llm = CFG.unipred_llm_loop and pred_config.get(
                "llm_proposals", True)
            self._proposers[curr_pred] = EffectProposer(
                client=self._llm() if use_llm else None,
                target=curr_pred,
                rows=self.ae_row_names,
                known_predicates=sorted(self._initial_predicates),
                names=self._names(),
                allowed=allowed,
                seeds=self._seed_vectors().get(curr_pred, []),
                use_llm=use_llm,
                loss_feedback=CFG.unipred_loss_feedback,
                description=self._description(),
                accept_loss=pred_config["loss_thresh"],
                zero_loss=CFG.unipred_score_zero_loss)
        return self._proposers[curr_pred]

    def _save_llm_history(self) -> None:
        """Save every proposal and its validation loss, for analysis."""
        history: Dict[str, List[Dict]] = {}
        for pred, proposer in self._proposers.items():
            entries = []
            for codes, loss in proposer.history.items():
                entries.append({
                    "add": [
                        o.name for o, c in zip(self.ae_row_names, codes)
                        if c == 1
                    ],
                    "del": [
                        o.name for o, c in zip(self.ae_row_names, codes)
                        if c == 2
                    ],
                    "val_loss": loss,
                })
            history[pred.name] = entries
        with open(f"{utils.get_approach_save_path_str()}.llm_history.json",
                  "w",
                  encoding="utf-8") as f:
            json.dump(history, f, indent=1)
