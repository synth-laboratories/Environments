#!/usr/bin/env bash
set -euo pipefail

# Quick parity check between pure-Python Sokoban and PyO3-backed Sokoban via the Python service
# - Ensures the PyO3 module is importable (builds it with maturin if needed)
# - Runs the parity smoke test script against the in-process FastAPI app
#
# Usage:
#   scripts/check_parity.sh
#
# Optional env vars:
#   PYTHON_BIN   Path to python executable (default: python3)
#   SKIP_BUILD   If set to 1, skip building the PyO3 module

PYTHON_BIN=${PYTHON_BIN:-python3}
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

PYVER="$(${PYTHON_BIN} -V 2>&1)"
echo "[Parity] Using Python: ${PYVER}"
MAJOR=$(${PYTHON_BIN} - <<'PY'
import sys
print(sys.version_info.major)
PY
)
MINOR=$(${PYTHON_BIN} - <<'PY'
import sys
print(sys.version_info.minor)
PY
)
if [[ ${MAJOR} -lt 3 || ( ${MAJOR} -eq 3 && ${MINOR} -lt 11 ) ]]; then
  echo "[Parity][ERROR] Python 3.11+ required to import horizons_env_py (built as abi3-py311)."
  echo "         Set PYTHON_BIN to a 3.11+ interpreter, e.g.: PYTHON_BIN=python3.11 scripts/check_parity.sh"
  exit 1
fi

# 1) Ensure horizons_env_py is importable, try building if not
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "[Parity] Checking horizons_env_py import..."
  if ! ${PYTHON_BIN} - <<'PY' >/dev/null 2>&1; then
import importlib
import sys
ok = True
try:
    importlib.import_module('horizons_env_py')
except Exception as e:
    ok = False
    print(e, file=sys.stderr)
sys.exit(0 if ok else 1)
PY
    echo "[Parity] horizons_env_py not importable; attempting maturin develop..."
    if ! command -v maturin >/dev/null 2>&1; then
      echo "[Parity][ERROR] maturin not found. Install it or run in a venv with maturin available."
      echo "  pip install maturin  (or: uv pip install maturin)"
      exit 1
    fi
    (cd "$ROOT_DIR/rust_port/horizons_env_py" && maturin develop)
    echo "[Parity] Built horizons_env_py via maturin."
  else
    echo "[Parity] horizons_env_py is importable."
  fi
else
  echo "[Parity] SKIP_BUILD=1 set; skipping horizons_env_py build check."
fi

# 2) Run the parity smoke test
echo "[Parity] Running parity_service_pyo3_vs_pure.py..."
set -x
${PYTHON_BIN} "$ROOT_DIR/scripts/parity_service_pyo3_vs_pure.py"
rc=$?
set +x

if [[ $rc -eq 0 ]]; then
  echo "[Parity] OK: PyO3 service and pure service agree on basic keys and termination."
else
  echo "[Parity] FAILED: see output above for mismatches."
fi

exit $rc
