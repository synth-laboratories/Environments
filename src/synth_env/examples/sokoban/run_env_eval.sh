#!/usr/bin/env bash
# Evaluation runner for Sokoban environment (placeholder).
set -euo pipefail

if [[ "${1:-}" == "--info" ]]; then
  echo "sokoban : --model_name <MODEL> --episodes <N> --max_steps <STEPS> [--seed <SEED>]"
  exit 0
fi

echo "[Sokoban] run_env_eval.sh is not implemented yet. Use --info for parameter schema." >&2
exit 1 