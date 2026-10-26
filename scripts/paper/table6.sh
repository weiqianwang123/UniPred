#!/usr/bin/env bash
# Table VI: ways of using the LLM for planning, in Blocks.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
experiment blocks unipred
experiment blocks unipred no_llm_loop
experiment blocks code_as_policy
summarize --title "Table VI" --domains blocks --runs \
    unipred="UniPred (Ours)" unipred+no_llm_loop="LLM complete PDDL" \
    code_as_policy="LLM propose code (Code as Policy)"
