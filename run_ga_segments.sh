#!/bin/bash

# Script to run genetic algorithm on multiple segments in parallel
# Usage: ./run_ga_segments.sh <track> <population_size> <generations> <base_record> <segments>
# Where <segments> is a space or comma-separated list of segment numbers, e.g., "1 2 3" or "1,2,3"

if [ $# -ne 5 ]; then
    echo "Usage: $0 <track> <population_size> <generations> <base_record> <segments>"
    echo "  <base_record>: Full path to the record directory, e.g., visual_records/record_0"
    echo "  <segments>: Space or comma-separated list of segment numbers, e.g., '1 2 3' or '1,2,3'"
    exit 1
fi

TRACK=$1
POP_SIZE=$2
GENS=$3
BASE_RECORD=$4
SEGMENTS_ARG=$5

# Parse segments: replace commas with spaces and split
SEGMENTS=$(echo "$SEGMENTS_ARG" | tr ',' ' ')

echo "Running genetic algorithm for track: $TRACK"
echo "Population size: $POP_SIZE, Generations: $GENS"
echo "Base record: $BASE_RECORD"
echo "Segments: $SEGMENTS"

# Run each segment in parallel
for SEGMENT in $SEGMENTS; do
    echo "Starting segment $SEGMENT..."
    python scripts/genetic_algorithm.py "$TRACK" "$POP_SIZE" "$GENS" "$SEGMENT" "$BASE_RECORD" &
done

# Wait for all background processes to complete
wait

echo "All segments completed."