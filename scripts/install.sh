#!/usr/bin/env bash
# Install UniPred from a fresh clone: the submodules, a conda environment
# with both packages (editable), Fast Downward and, with --weights, the
# perception weights.
#
#   scripts/install.sh            # environment, packages, Fast Downward
#   scripts/install.sh --weights  # also GroundingDINO, SAM2 and DINOv3
#
# ENV_NAME names the conda environment (default: unipred). The weights go to
# $UNIPRED_WEIGHTS_DIR (default: weights/ in the repository), one directory
# per model (unipred/perception/weights.py). DINOv3 is gated on Hugging Face:
# accept its licence on the model page and log in (hf auth login) first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo=$PWD
env_name=${ENV_NAME:-unipred}
weights_dir=${UNIPRED_WEIGHTS_DIR:-$repo/weights}

git submodule update --init --recursive

if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda create -y -n "$env_name" python=3.10
fi
run() { conda run --no-capture-output -n "$env_name" "$@"; }
run pip install -e third_party/IVNTR
run pip install -e ".[dev,perception]"

(cd third_party/IVNTR && run scripts/build_downward.sh)

if [[ ${1:-} == --weights ]]; then
    mkdir -p "$weights_dir"
    run python - "$weights_dir" <<'PY'
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
    "sam2": "facebook/sam2.1-hiera-tiny",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
for name, repo_id in MODELS.items():
    print(f"Downloading {repo_id} to {sys.argv[1]}/{name}")
    snapshot_download(repo_id, local_dir=f"{sys.argv[1]}/{name}")
PY
fi

cat <<EOF
UniPred is installed. Before running it:
  conda activate $env_name
  export FD_EXEC_PATH=$repo/third_party/IVNTR/ext/downward
  export UNIPRED_WEIGHTS_DIR=$weights_dir   # for the perception weights
EOF
