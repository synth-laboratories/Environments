#!/usr/bin/env bash
# Evaluation runner for Verilog environment (placeholder).
set -euo pipefail

if [[ "${1:-}" == "--info" ]]; then
  echo "verilog : --model_name <MODEL> --suite <SUITE> --episodes <N> [--seed <SEED>]"
  exit 0
fi

echo "[Verilog] run_env_eval.sh is not implemented yet. Use --info for parameter schema." >&2
exit 1 