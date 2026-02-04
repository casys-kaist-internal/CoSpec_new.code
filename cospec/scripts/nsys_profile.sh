#!/bin/bash
# nsys hardware metrics profiling
# Usage:
#   ./nsys_profile.sh              - profile infinite while loop (baseline)
#   ./nsys_profile.sh <command>    - profile the given command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/profile"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ $# -eq 0 ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/hw_metrics_baseline_${TIMESTAMP}"
    COMMAND='python3 -c "while True: pass"'
else
    OUTPUT_FILE="${OUTPUT_DIR}/hw_metrics_${TIMESTAMP}"
    COMMAND="$@"
fi

echo "Output will be saved to: ${OUTPUT_FILE}.nsys-rep"
echo "Command: $COMMAND"
echo "Press Ctrl+C to stop profiling"

nsys profile \
    --output="$OUTPUT_FILE" \
    --gpu-metrics-device=0 \
    --gpu-metrics-frequency=10000 \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    $COMMAND
