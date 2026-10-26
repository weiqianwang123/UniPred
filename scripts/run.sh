#!/usr/bin/env bash
# Run one experiment: train, test on the out-of-distribution test tasks,
# then test the saved model on in-distribution test tasks.
#
#   scripts/run.sh CONFIG SEED [FLAGS...]
#   scripts/run.sh configs/satellites/unipred.yaml 0
#   LLM=openai scripts/run.sh configs/satellites/unipred.yaml 0 \
#       --config configs/ablations/no_loss_feedback.yaml
#
# LLM selects configs/llm/<LLM>.yaml: codex (the local Codex CLI, default),
# openai, qwen, gemini or mock. Outputs go to outputs/<domain>/<config>/
# seed<SEED>/ (OUT overrides the directory). PYTHON overrides the
# interpreter (default: python of the active environment). FD_EXEC_PATH
# defaults to the Fast Downward submodule of the IVNTR fork.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FD_EXEC_PATH=${FD_EXEC_PATH:-$PWD/third_party/IVNTR/ext/downward}

config=$1
seed=$2
shift 2
llm_config="configs/llm/${LLM:-codex}.yaml"
python=${PYTHON:-python}
domain=$(basename "$(dirname "$config")")
out=${OUT:-outputs/$domain/$(basename "$config" .yaml)/seed$seed}
common=(--config "$config" --config "$llm_config" --seed "$seed"
        --approach_dir "$out/approach" --neupi_save_path "$out/approach"
        --data_dir outputs/datasets --llm_cache_dir outputs/llm_cache)

"$python" -m unipred.main "${common[@]}" --results_dir "$out/results" \
    --eval_trajectories_dir "$out/eval" --log_file "$out/train_test.log" "$@"
"$python" -m unipred.main "${common[@]}" --load_approach --in_domain_test True \
    --results_dir "$out/results_in_domain" \
    --eval_trajectories_dir "$out/eval_in_domain" \
    --log_file "$out/test_in_domain.log" "$@"
