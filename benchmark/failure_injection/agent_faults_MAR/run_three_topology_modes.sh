#!/usr/bin/env bash
# Sample tasks once, then run edge-preference, cloud-preference, and dynamic.
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$dir/../../.." && pwd)"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
exec "$python_bin" "$dir/../run_three_topology_modes.py" "$(basename "$dir")" "$@"
