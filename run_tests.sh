#!/usr/bin/env bash
# Run the full sakz-bot test suite. Exits non-zero if any test fails.
# Each test_*.py uses proper exit codes (sys.exit / SystemExit / unittest).
set -uo pipefail

cd "$(dirname "$0")"

TESTS=(
  test_signal_behavior.py
  test_autoscan_lifecycle.py
  test_prime.py
  test_prime_integration.py
  test_phase0.py
  test_wiring.py
  test_scale_hardening.py
  test_security_fixes.py
  test_orders.py
)

fail=0
for t in "${TESTS[@]}"; do
  if [[ ! -f "$t" ]]; then
    echo "SKIP  $t (not found)"
    continue
  fi
  echo "==================== $t ===================="
  if python3 "$t"; then
    echo "PASS  $t"
  else
    echo "FAIL  $t"
    fail=1
  fi
  echo
done

if [[ "$fail" -ne 0 ]]; then
  echo "RESULT: one or more test files FAILED"
  exit 1
fi
echo "RESULT: all test files passed"
