#!/usr/bin/env bash
set -euo pipefail

# Create a local .venv using Python 3.11 and install the project + PyO3 bridge
# Usage: bash scripts/setup_py311_venv.sh

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "[setup] Python 3.11 not found. Install 3.11 (e.g., via pyenv) and rerun." >&2
  exit 1
fi

PY=python3.11
echo "[setup] Using $(${PY} -V)"

echo "[setup] Creating .venv with Python 3.11..."
${PY} -m venv .venv
echo "[setup] .venv created. To activate: source .venv/bin/activate"

echo "[setup] Installing project in editable mode..."
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

echo "[setup] Ensuring maturin is available and building PyO3 bridge..."
python -m pip install -U maturin
(cd rust_port/horizons_env_py && maturin develop)

echo "[setup] Done. You can now run:"
echo "       - uvicorn horizons.environments.service.app:app --reload --port 8901"
echo "       - bash scripts/check_parity.sh"

