#!/usr/bin/env bash
set -u

# Run MiniGrid Python and Rust tests and print a summary table

ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
PYTEST_PATH="rust_port/vendored_envs/minigrid/validity"
CRATE_DIR="rust_port/vendored_envs/minigrid"

TS="$(date +%s)"
OUTDIR="${TMPDIR:-/tmp}/minigrid_test_$TS"
mkdir -p "$OUTDIR"

PY_OUT="$OUTDIR/pytest.out"
RS_OUT="$OUTDIR/cargo.out"

echo "[MiniGrid] Running pytest on $PYTEST_PATH"
set +e
(
  cd "$ROOT" && pytest "$PYTEST_PATH" -q
) | tee "$PY_OUT"
PY_RC=${PIPESTATUS[0]:-0}
set -e

echo "[MiniGrid] Running cargo test in $CRATE_DIR"
set +e
(
  cd "$ROOT/$CRATE_DIR" && cargo test -q
) | tee "$RS_OUT"
RS_RC=${PIPESTATUS[0]:-0}
set -e

# --- Parse pytest results ---
PY_PASSED=0; PY_FAILED=0; PY_SKIPPED=0; PY_STATUS="OK"
if [ -s "$PY_OUT" ]; then
  SUM_LINE=$(grep -E "[0-9]+ (passed|failed|skipped)" "$PY_OUT" | tail -n 1 || true)
  if [[ -n "$SUM_LINE" ]]; then
    if echo "$SUM_LINE" | grep -qE "([0-9]+) passed"; then PY_PASSED=$(echo "$SUM_LINE" | sed -n 's/.*\([0-9][0-9]*\) passed.*/\1/p'); fi
    if echo "$SUM_LINE" | grep -qE "([0-9]+) failed"; then PY_FAILED=$(echo "$SUM_LINE" | sed -n 's/.*\([0-9][0-9]*\) failed.*/\1/p'); fi
    if echo "$SUM_LINE" | grep -qE "([0-9]+) skipped"; then PY_SKIPPED=$(echo "$SUM_LINE" | sed -n 's/.*\([0-9][0-9]*\) skipped.*/\1/p'); fi
  fi
fi
if [ $PY_RC -ne 0 ] || [ "$PY_FAILED" -gt 0 ]; then PY_STATUS="FAIL"; fi

# --- Parse cargo results ---
RS_PASSED=0; RS_FAILED=0; RS_IGNORED=0; RS_STATUS="OK"
if [ -s "$RS_OUT" ]; then
  # Sum across all "test result:" lines
  while IFS= read -r line; do
    case "$line" in
      *"test result:"*)
        p=$(echo "$line" | sed -n 's/.*\([0-9][0-9]*\) passed.*/\1/p')
        f=$(echo "$line" | sed -n 's/.*\([0-9][0-9]*\) failed.*/\1/p')
        i=$(echo "$line" | sed -n 's/.*\([0-9][0-9]*\) ignored.*/\1/p')
        [ -n "$p" ] && RS_PASSED=$((RS_PASSED + p))
        [ -n "$f" ] && RS_FAILED=$((RS_FAILED + f))
        [ -n "$i" ] && RS_IGNORED=$((RS_IGNORED + i))
        ;;
    esac
  done < "$RS_OUT"
fi
if [ $RS_RC -ne 0 ] || [ "$RS_FAILED" -gt 0 ]; then RS_STATUS="FAIL"; fi

# --- Print summary table ---
printf "\n\nMiniGrid Test Summary\n"
printf "====================\n"
printf "%-18s | %7s | %6s | %7s | %6s\n" "Suite" "Passed" "Failed" "Skipped" "Status"
printf "%s\n" "--------------------+---------+--------+---------+--------"
printf "%-18s | %7d | %6d | %7d | %6s\n" "pytest(validity)" "$PY_PASSED" "$PY_FAILED" "$PY_SKIPPED" "$PY_STATUS"
printf "%-18s | %7d | %6d | %7d | %6s\n" "cargo(minigrid-rs)" "$RS_PASSED" "$RS_FAILED" "$RS_IGNORED" "$RS_STATUS"

echo "\nDetails saved in $OUTDIR"

exit $(( (PY_RC != 0) || (RS_RC != 0) ))
