#!/usr/bin/env bash
# Table VII: UniPred with and without the names of operators and known
# predicates in the prompts, and IVNTR, averaged over Satellites and Blocks
# (the paper's third domain, Tools, is not part of this release).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
domains=(satellites blocks)
for domain in "${domains[@]}"; do
    experiment "$domain" unipred
    experiment "$domain" unipred no_semantic
    experiment "$domain" ivntr
done
summarize --title "Table VII" --average --domains "${domains[@]}" --runs \
    unipred="UniPred (Ours)" unipred+no_semantic="UniPred w/o Semantic" \
    ivntr=IVNTR
