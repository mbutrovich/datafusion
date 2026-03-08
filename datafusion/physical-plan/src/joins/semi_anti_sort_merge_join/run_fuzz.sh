#!/bin/bash
set -euo pipefail

cd /Users/matt/git/datafusion

export RUSTFLAGS="-C debug-assertions=no"

# Each filter substring matches both the i32 and binary variants
TESTS=(
  "join_fuzz::test_left_semi_join_1k"
  "join_fuzz::test_left_anti_join_1k"
  "join_fuzz::test_right_semi_join_1k"
  "join_fuzz::test_right_anti_join_1k"
)

ITERATIONS=${1:-10}
FAILS=0
TOTAL=0

echo "Starting at $(date)"
echo "Running $ITERATIONS iterations of ${#TESTS[@]} test groups"

for i in $(seq 1 "$ITERATIONS"); do
  for t in "${TESTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    OUTPUT=$(cargo test --features extended_tests -p datafusion --test fuzz -- "$t" 2>&1 || true)

    # Check for failure: "FAILED" in test result line
    HAS_FAIL=$(echo "$OUTPUT" | grep -c "FAILED" || true)
    # Check tests actually ran: "running N test" where N > 0
    HAS_RUN=$(echo "$OUTPUT" | grep -c "running [1-9]" || true)

    if [ "$HAS_RUN" -eq 0 ]; then
      echo "ERROR: no tests ran for $t at iteration $i"
      echo "$OUTPUT" | tail -5
      exit 1
    elif [ "$HAS_FAIL" -gt 0 ]; then
      echo "FAIL: $t iteration $i"
      echo "--- full output ---"
      echo "$OUTPUT"
      echo "--- end output ---"
      exit 1
    fi
  done
  echo "iteration $i done at $(date) (total: $TOTAL, fails: $FAILS)"
done

echo "DONE at $(date): $FAILS / $TOTAL failed"
