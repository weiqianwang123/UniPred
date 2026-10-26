#!/usr/bin/env bash
# Install UniPred from a fresh clone: the submodules, a conda environment
# with both packages (editable), Fast Downward and, with --weights, the
# perception weights; with --sim, MuJoCo and the simulated robot's assets.
#
#   scripts/install.sh                  # environment, packages, Fast Downward
#   scripts/install.sh --weights        # also GroundingDINO, SAM2 and DINOv3
#   scripts/install.sh --weights --sim  # also the simulated robot
#
# ENV_NAME names the conda environment (default: unipred). The weights go to
# $UNIPRED_WEIGHTS_DIR (default: weights/ in the repository), one directory
# per model (unipred/perception/weights.py). DINOv3 is gated on Hugging Face:
# accept its licence on the model page and log in (hf auth login) first.
# The simulator's assets go to $UNIPRED_SIM_ASSETS_DIR (default: sim_assets/
# in the repository; unipred/robot_sim/assets.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo=$PWD
env_name=${ENV_NAME:-unipred}
weights_dir=${UNIPRED_WEIGHTS_DIR:-$repo/weights}
weights=false sim=false
for arg in "$@"; do
    case $arg in
        --weights) weights=true ;;
        --sim) sim=true ;;
        *) echo "Unknown option $arg" >&2; exit 2 ;;
    esac
done

git submodule update --init --recursive

if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda create -y -n "$env_name" python=3.10
fi
run() { conda run --no-capture-output -n "$env_name" "$@"; }
run pip install -e third_party/IVNTR
run pip install -e ".[dev,perception]"

(cd third_party/IVNTR && run scripts/build_downward.sh)

if $sim; then
    run pip install -e ".[sim]"
    run python -m unipred.robot_sim.assets
fi

if $weights; then
    mkdir -p "$weights_dir"
    run python - "$weights_dir" <<'PY'
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

MODELS = {
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
    "sam2": "facebook/sam2.1-hiera-tiny",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
weights_dir = Path(sys.argv[1])
failed = []
for name, repo_id in MODELS.items():
    target = weights_dir / name
    if (target / "config.json").exists():
        print(f"{repo_id} is in {target}")
        continue
    print(f"Downloading {repo_id} to {target}")
    # Download next to the target and move it there when complete.
    tmp = Path(tempfile.mkdtemp(dir=weights_dir))
    try:
        # transformers loads the safetensors; skip the other formats.
        snapshot_download(repo_id, local_dir=tmp,
                          ignore_patterns=["*.bin", "*.pt", "*.pth"])
    except GatedRepoError:
        print(f"{repo_id} is gated: accept its licence on "
              f"https://huggingface.co/{repo_id}, log in with "
              "`hf auth login`, and run this again.")
        failed.append(repo_id)
        shutil.rmtree(tmp)
        continue
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
if failed:
    sys.exit(f"Not downloaded: {', '.join(failed)}")
PY
fi

cat <<EOF
UniPred is installed. Before running it:
  conda activate $env_name
  export FD_EXEC_PATH=$repo/third_party/IVNTR/ext/downward
  export UNIPRED_WEIGHTS_DIR=$weights_dir   # for the perception weights
EOF
