#!/bin/bash
# Compress the results folder with a timestamp

RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="results_${TIMESTAMP}.tar.gz"

if [ -d "$RESULTS_DIR" ]; then
    tar -czf "$ARCHIVE_NAME" "$RESULTS_DIR"
    echo "✅ Created: $ARCHIVE_NAME ($(du -h "$ARCHIVE_NAME" | cut -f1))"
else
    echo "❌ Error: '$RESULTS_DIR' directory not found."
    exit 1
fi
