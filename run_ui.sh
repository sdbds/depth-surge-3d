#!/bin/bash

echo "Starting Depth Surge 3D Web UI..."

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Follow docs/INSTALLATION.md first."
    exit 1
fi

# Create output directory
mkdir -p output

# Start the Flask application
echo "Starting web server at http://localhost:5000"
echo "Navigate to http://localhost:5000"
echo "Press Ctrl+C to stop this server"

# Open browser automatically (cross-platform)
if command -v xdg-open > /dev/null; then
    echo "Opening in existing browser session."
    (sleep 2; xdg-open http://localhost:5000) &
elif command -v open > /dev/null; then
    echo "Opening in default browser."
    (sleep 2; open http://localhost:5000) &
elif command -v start > /dev/null; then
    echo "Opening in default browser."
    (sleep 2; start http://localhost:5000) &
fi

# The app now receives signals directly, and its exit status is the launcher's.
export DEPTH_SURGE_UI_SCRIPT=1
exec python app.py "$@"
