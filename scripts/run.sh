#!/usr/bin/env bash
# Run one experiment: train, test on the out-of-distribution test tasks,
# then test the saved model on in-distribution test tasks.
#
#   scripts/run.sh CONFIG SEED [FLAGS...]
#   scripts/run.sh configs/satellites/unipred.yaml 0
#   LLM=openai scripts/run.sh configs/satellites/unipred.yaml 0 \
#       --config configs/ablations/no_loss_feedback.yaml
#
# LLM selects configs/llm/<LLM>.yaml: codex (the local Codex CLI; the
# default if it is installed), openai (the default otherwise), qwen, gemini
# or mock. Outputs go to outputs/<domain>/<run>/seed<SEED>/, where <run> is
# the config's name followed by +<name> for each --config overlay among the
# flags (e.g. unipred+no_loss_feedback); OUT overrides the directory.
# PYTHON overrides the interpreter (default: python of the active
# environment). FD_EXEC_PATH defaults to the Fast Downward submodule of the
# IVNTR fork.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FD_EXEC_PATH=${FD_EXEC_PATH:-$PWD/third_party/IVNTR/ext/downward}

config=$1
seed=$2
shift 2
if [[ -z ${LLM:-} ]]; then
    if command -v codex > /dev/null; then LLM=codex; else LLM=openai; fi
fi
llm_config="configs/llm/$LLM.yaml"
python=${PYTHON:-python}
domain=$(basename "$(dirname "$config")")
run=$(basename "$config" .yaml)
flags=("$@")
for ((i = 0; i + 1 < ${#flags[@]}; i++)); do
    if [[ ${flags[i]} == --config ]]; then
        run+="+$(basename "${flags[i + 1]}" .yaml)"
    fi
done
out=${OUT:-outputs/$domain/$run/seed$seed}
common=(--config "$config" --config "$llm_config" --seed "$seed"
        --approach_dir "$out/approach" --neupi_save_path "$out/approach"
        --data_dir outputs/datasets --llm_cache_dir outputs/llm_cache)

"$python" -m unipred.main "${common[@]}" --results_dir "$out/results" \
    --eval_trajectories_dir "$out/eval" --log_file "$out/train_test.log" "$@"
"$python" -m unipred.main "${common[@]}" --load_approach --in_domain_test True \
    --results_dir "$out/results_in_domain" \
    --eval_trajectories_dir "$out/eval_in_domain" \
    --log_file "$out/test_in_domain.log" "$@"
