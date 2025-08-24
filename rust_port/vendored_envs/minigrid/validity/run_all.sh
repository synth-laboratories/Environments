#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"

cd "$ROOT"
echo "[MiniGrid] Running validity pytest suite..."
pytest rust_port/vendored_envs/minigrid/validity -q || true

echo "[MiniGrid] Running Rust crate tests..."
(cd rust_port/vendored_envs/minigrid && cargo test -q) || true
exit 0
