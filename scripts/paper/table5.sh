#!/usr/bin/env bash
# Table V: the stages of unified bilevel learning, averaged over the
# simulated domains without derived predicates (Tools is not part of this
# release).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
domains=(satellites blocks packing)
for domain in "${domains[@]}"; do
    experiment "$domain" unipred
    for overlay in no_pddl_completion no_llm_loop no_loss_feedback; do
        experiment "$domain" unipred "$overlay"
    done
done
summarize --title "Table V" --average --domains "${domains[@]}" --runs \
    unipred="UniPred (Ours)" \
    unipred+no_pddl_completion="UniPred w/o PDDL Completion" \
    unipred+no_llm_loop="UniPred w/o Unified Bilevel Learning" \
    unipred+no_loss_feedback="UniPred w/o Loss Feedback"
