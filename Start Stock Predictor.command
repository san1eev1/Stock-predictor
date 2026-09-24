#!/bin/bash
# Double-click this file in Finder to start the Stock Predictor.
# It opens the web dashboard in your browser and runs the live monitor.
# Close this window or press Ctrl+C to stop.
cd "$(dirname "$0")" || exit 1
echo "Updating code..."
git pull --quiet || echo "(could not update - continuing with the current version)"
source .venv/bin/activate || { echo "Run the '1. First-time setup' task in VS Code first."; read -r; exit 1; }
pip install --quiet -e . >/dev/null 2>&1
caffeinate -i python -m stockpredictor start
