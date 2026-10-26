#!/usr/bin/env bash
# Table II: success rates and learning time of every method on the
# simulated domains (Tools is not part of this release).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
domains=(satellites blocks packing table_clean_sim)
for domain in "${domains[@]}"; do
    for config in oracle unipred ivntr gnn transformer grammar random; do
        experiment "$domain" "$config"
    done
    experiment "$domain" unipred llm_qwen
    experiment "$domain" unipred llm_gemini
done
summarize --title "Table II" --domains "${domains[@]}" --runs \
    oracle=Oracle unipred="UniPred (Ours)" ivntr=IVNTR gnn=GNN \
    transformer=Transformer grammar=Grammar random=Random \
    unipred+llm_qwen="UniPred(Qwen)" unipred+llm_gemini="UniPred(Gemini)"
