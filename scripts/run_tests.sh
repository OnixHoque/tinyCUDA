#!/usr/bin/env bash
set -e

TEST_DIR="tests"
SCRIPT_DIR="$(dirname "$0")"

echo "[INFO] Running all tests in $TEST_DIR ..."

for test_file in "$TEST_DIR"/*.cu; do
    test_name=$(basename "$test_file" .cu)
    echo "--------------------------------------"
    echo "[INFO] Building and running $test_name ..."
    
    # Use build_and_run.sh
    "$SCRIPT_DIR/build_and_run.sh" "$test_file" "$test_name"
    
    echo "[PASS] $test_name succeeded"
done

echo "======================================"
echo "[PASS] All tests completed successfully"
