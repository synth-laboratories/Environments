#!/usr/bin/env bash
# Top-level umbrella script for running environment evaluations.
# Minimal v0: only implements `--show` which lists supported envs and their CLI params.

set -euo pipefail

# Directory of the repository root (where this script lives)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hard-coded list of env helper scripts for v0. Each path is relative to $ROOT_DIR.
ENV_SCRIPTS=(
  "src/synth_env/examples/crafter_classic/run_env_eval.sh"
  "src/synth_env/examples/sokoban/run_env_eval.sh"
  "src/synth_env/examples/verilog/run_env_eval.sh"
)

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--show] | <ENV_NAME> [env-specific args]

Examples:
  # Show all supported environments and their parameter schema
  $(basename "$0") --show

  # Run a Crafter evaluation
  $(basename "$0") crafter --model_name gpt-4o-mini --episodes 10 --max_steps 400
EOF
}

if [[ "$#" -eq 0 ]]; then
  print_usage
  exit 0
fi

if [[ "$1" == "--config" ]]; then
  CONFIG_PATH="$2"; shift 2
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
  fi
  # Use Python (via uv) to parse TOML and spit out ENV_NAME and ARGS string
  mapfile -t PARSED < <(uv run -- python - <<'PY' "$CONFIG_PATH"
import sys, tomllib, shlex
cfg_path = sys.argv[1]
with open(cfg_path, 'rb') as f:
    data = tomllib.load(f)
if 'eval' not in data:
    raise SystemExit('TOML missing [eval] section')
conf = data['eval']
if 'env' not in conf:
    raise SystemExit('TOML missing env field in [eval]')
env_name = conf.pop('env')
# Build arg list
arg_parts = []
for k, v in conf.items():
    if isinstance(v, bool):
        if v:
            arg_parts.append(f"--{k}")
    else:
        arg_parts.extend([f"--{k}", str(v)])
print(env_name)
print(' '.join(shlex.quote(p) for p in arg_parts))
PY)
  ENV_NAME="${PARSED[0]}"
  EXTRA_ARGS="${PARSED[1]}"
  # Recursively call self with computed env and args
  exec "$0" "$ENV_NAME" ${EXTRA_ARGS}
fi

case "$1" in
  -h|--help)
    print_usage
    exit 0
    ;;
  --show)
    echo "Supported evaluation environments:"
    echo "-----------------------------------"
    for rel_path in "${ENV_SCRIPTS[@]}"; do
      script_path="$ROOT_DIR/$rel_path"
      if [[ -x "$script_path" ]]; then
        "$script_path" --info || true
      else
        env_name="$(basename "$(dirname "$script_path")")"
        echo "$env_name : helper script not found (expected at $rel_path)"
      fi
    done
    exit 0
    ;;
  *)
    ENV_NAME="$1"; shift
    # Map ENV_NAME to a helper script path (crudely for v0)
    case "$ENV_NAME" in
      crafter)
        HELPER="src/synth_env/examples/crafter_classic/run_env_eval.sh";;
      sokoban)
        HELPER="src/synth_env/examples/sokoban/run_env_eval.sh";;
      verilog)
        HELPER="src/synth_env/examples/verilog/run_env_eval.sh";;
      *)
        echo "Unknown environment: $ENV_NAME" >&2
        exit 1
        ;;
    esac

    HELPER_PATH="$ROOT_DIR/$HELPER"
    if [[ ! -x "$HELPER_PATH" ]]; then
      echo "Helper script not found or not executable: $HELPER_PATH" >&2
      exit 1
    fi

    "$HELPER_PATH" "$@"
    ;;
esac
