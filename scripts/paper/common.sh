# Shared by the table scripts (sourced): run the experiments a table needs
# with scripts/run.sh, then print the table with unipred.eval.summarize.
#
#   SEEDS  the seeds to run (default: the paper's five, "0 1 2 3 4")
#   RUN=0  only print the table from the results that exist
#
# Seeds that already have results are not run again. LLM selects the LLM
# as in scripts/run.sh, and PYTHON the interpreter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SEEDS=${SEEDS:-0 1 2 3 4}

# experiment DOMAIN CONFIG [OVERLAY...]: every seed of configs/DOMAIN/
# CONFIG.yaml with the overlays configs/ablations/OVERLAY.yaml.
experiment() {
    local domain=$1 config=$2
    shift 2
    local run=$config flags=()
    for overlay in "$@"; do
        run+="+$overlay"
        flags+=(--config "configs/ablations/$overlay.yaml")
    done
    for seed in $SEEDS; do
        if compgen -G "outputs/$domain/$run/seed$seed/results_in_domain/*.pkl" \
            > /dev/null || [[ ${RUN:-1} == 0 ]]; then
            continue
        fi
        scripts/run.sh "configs/$domain/$config.yaml" "$seed" "${flags[@]}"
    done
}

summarize() {
    "${PYTHON:-python}" -m unipred.eval.summarize "$@"
}
