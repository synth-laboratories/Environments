#!/usr/bin/env bash
# Evaluation runner for Crafter environment (placeholder).
set -euo pipefail

if [[ "${1:-}" == "--info" ]]; then
  echo "crafter : --model_name <MODEL> --episodes <N> --max_steps <STEPS> [--seed <SEED>]"
  exit 0
fi

echo "[Crafter] run_env_eval.sh is not implemented yet. Use --info for parameter schema." >&2
exit 1 