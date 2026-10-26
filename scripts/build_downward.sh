#!/usr/bin/env bash
# Build the Fast Downward planner from the ext/downward git submodule.
#
# Usage: scripts/build_downward.sh [JOBS]   (default: 8 parallel jobs)
# Then point the planners in predicators/planning.py at the build:
#   export FD_EXEC_PATH=<path to this repo>/ext/downward
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fd_dir="${repo_root}/ext/downward"
jobs="${1:-8}"

if [ ! -f "${fd_dir}/build.py" ]; then
    git -C "${repo_root}" submodule update --init ext/downward
fi
cd "${fd_dir}"
python3 build.py "-j${jobs}"
echo "Fast Downward is built. Use: export FD_EXEC_PATH=${fd_dir}"
