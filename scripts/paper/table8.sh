#!/usr/bin/env bash
# Table VIII: derived-aware predicate selection, in Table Clean Sim.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
experiment table_clean_sim unipred
experiment table_clean_sim ivntr
experiment table_clean_sim unipred no_derived_selection
experiment table_clean_sim ivntr_derived
summarize --title "Table VIII" --domains table_clean_sim --runs \
    unipred="UniPred (Ours)" ivntr=IVNTR \
    unipred+no_derived_selection="UniPred w/o derived. select" \
    ivntr_derived="IVNTR w/ derived. select"
