#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/qianwei/UniPred}"
CONDA_ENV="${CONDA_ENV:-ivntr}"
SEED="${SEED:-0}"
BASE_CONFIG="${BASE_CONFIG:-predicators/config/satellites/pred_pdlm.yaml}"
PDDL_CONFIG="${PDDL_CONFIG:-predicators/config/satellites/pddl.json}"
SWEEP_DIR="${SWEEP_DIR:-logs/satellites/unipred_num_vectors_sweep}"
DOMAIN_AAAI_THRESH="${DOMAIN_AAAI_THRESH:-300000}"

cd "$ROOT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export FD_EXEC_PATH="${FD_EXEC_PATH:-ext/downward}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"

CONFIG_DIR="$SWEEP_DIR/configs"
mkdir -p "$CONFIG_DIR" "$SWEEP_DIR" "saved_approaches/demo/satellites"

SUMMARY="$SWEEP_DIR/summary_seed_${SEED}.tsv"
printf "delta\tvector_counts\tconfig\tapproach_dir\ttrain_seconds\ttrain_status\tood_solved\tood_total\tindomain_seconds\tindomain_status\tindomain_solved\tindomain_total\n" > "$SUMMARY"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$SWEEP_DIR/gpu.txt" || true
fi

make_config() {
    local delta="$1"
    local out_config="$2"
    python3 - "$BASE_CONFIG" "$delta" "$out_config" <<'PY'
import pathlib
import sys

import yaml

base_config = pathlib.Path(sys.argv[1])
delta = int(sys.argv[2])
out_config = pathlib.Path(sys.argv[3])

with base_config.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

vector_counts = []
for block in data.get("config", []):
    if not isinstance(block, dict):
        continue
    if "num_vectors_to_generate" not in block:
        continue
    new_count = int(block["num_vectors_to_generate"]) + delta
    block["num_vectors_to_generate"] = new_count
    vector_counts.append(new_count)

for block in data.get("config", []):
    if isinstance(block, dict) and block.get("name") == "other":
        if "num_vectors_to_generate_list" in block:
            block["num_vectors_to_generate_list"] = vector_counts
        break

out_config.parent.mkdir(parents=True, exist_ok=True)
with out_config.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print(",".join(str(v) for v in vector_counts))
PY
}

extract_solved() {
    local log_file="$1"
    python3 - "$log_file" <<'PY'
import pathlib
import re
import sys

log_file = pathlib.Path(sys.argv[1])
if not log_file.exists():
    print("NA\tNA")
    raise SystemExit

text = log_file.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"Tasks solved:\s*(\d+)\s*/\s*(\d+)", text)
if not matches:
    print("NA\tNA")
else:
    solved, total = matches[-1]
    print(f"{solved}\t{total}")
PY
}

run_train() {
    local delta="$1"
    local config="$2"
    local approach_dir="$3"
    local train_log="$4"
    local train_stdout="$5"

    local start_time end_time status
    start_time=$(date +%s)
    set +e
    python3 predicators/main.py --env satellites --approach unipred \
        --seed "$SEED" --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --neupi_pred_config "$config" \
        --pred_pddl_config "$PDDL_CONFIG" \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --exclude_domain_feat "none" \
        --neupi_do_normalization False \
        --num_train_tasks 500 \
        --domain_aaai_thresh "$DOMAIN_AAAI_THRESH" \
        --neupi_entropy_w 0.5 \
        --neupi_loss_w 0.5 \
        --neupi_equ_dataset 1.0 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --sesame_max_samples_per_step 30 \
        --timeout 5 \
        --approach_dir "$approach_dir" \
        --neupi_save_path "$approach_dir" \
        --log_file "$train_log" > "$train_stdout" 2>&1
    status=$?
    set -e
    end_time=$(date +%s)

    echo "$((end_time - start_time))	$status"
}

run_indomain_test() {
    local config="$1"
    local approach_dir="$2"
    local indomain_log="$3"
    local indomain_stdout="$4"

    local start_time end_time status
    start_time=$(date +%s)
    set +e
    python3 predicators/main.py --env satellites --approach unipred \
        --seed "$SEED" --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --neupi_pred_config "$config" \
        --pred_pddl_config "$PDDL_CONFIG" \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --exclude_domain_feat "none" \
        --neupi_do_normalization False \
        --num_train_tasks 500 \
        --neupi_entropy_w 0.5 \
        --neupi_loss_w 0.5 \
        --load_data \
        --neupi_equ_dataset 1.0 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --execution_monitor expected_atoms \
        --load_approach \
        --load_neupi_from_json False \
        --in_domain_test True \
        --timeout 5 \
        --approach_dir "$approach_dir" \
        --neupi_load_pretrained "$approach_dir" \
        --log_file "$indomain_log" > "$indomain_stdout" 2>&1
    status=$?
    set -e
    end_time=$(date +%s)

    echo "$((end_time - start_time))	$status"
}

for delta in 0 1 2 3; do
    config="$CONFIG_DIR/pred_pdlm_num_vectors_plus_${delta}.yaml"
    vector_counts=$(make_config "$delta" "$config")
    approach_dir="saved_approaches/demo/satellites/unipred_num_vectors_plus_${delta}_${SEED}"
    train_log="$SWEEP_DIR/delta_${delta}_train.log"
    train_stdout="$SWEEP_DIR/delta_${delta}_train.stdout"
    indomain_log="$SWEEP_DIR/delta_${delta}_indomain.log"
    indomain_stdout="$SWEEP_DIR/delta_${delta}_indomain.stdout"

    echo "Running delta ${delta} with vector counts [${vector_counts}]"
    train_result=$(run_train "$delta" "$config" "$approach_dir" "$train_log" "$train_stdout")
    train_seconds=$(cut -f1 <<< "$train_result")
    train_status=$(cut -f2 <<< "$train_result")
    read -r ood_solved ood_total <<< "$(extract_solved "$train_log")"

    indomain_seconds="NA"
    indomain_status="SKIP"
    indomain_solved="NA"
    indomain_total="NA"
    if [[ "$train_status" == "0" ]]; then
        indomain_result=$(run_indomain_test "$config" "$approach_dir" "$indomain_log" "$indomain_stdout")
        indomain_seconds=$(cut -f1 <<< "$indomain_result")
        indomain_status=$(cut -f2 <<< "$indomain_result")
        read -r indomain_solved indomain_total <<< "$(extract_solved "$indomain_log")"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$delta" "$vector_counts" "$config" "$approach_dir" \
        "$train_seconds" "$train_status" "$ood_solved" "$ood_total" \
        "$indomain_seconds" "$indomain_status" "$indomain_solved" "$indomain_total" >> "$SUMMARY"

    echo "Finished delta ${delta}: train_status=${train_status}, ood=${ood_solved}/${ood_total}, indomain=${indomain_solved}/${indomain_total}"
done

echo "Sweep summary written to $SUMMARY"
