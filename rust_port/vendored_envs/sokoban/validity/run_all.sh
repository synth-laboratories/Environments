#!/usr/bin/env bash
set -euo pipefail

# Resolve to repo-absolute paths based on this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "[1/2] Running Python validity suite (pytest)"
(
  cd "$REPO_ROOT"
  # Activate local venv if present so horizons_env_py is available
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  pytest -q "$SCRIPT_DIR" -vv
)

echo "[2/2] Running Rust validity tests (cargo)"
(
  cd "$REPO_ROOT/rust_port/vendored_envs/sokoban"
  cargo test --quiet
)

echo "All validity tests passed."
