#!/bin/bash
# Quick test script to start UI and check for errors

echo "=== Starting UI Dashboard ==="
cd /Users/jaroncabral/Documents/Personal-Projects/flare-plus

# Make sure docker is running
if ! docker-compose ps > /dev/null 2>&1; then
    echo "Starting Docker services..."
    ./flare up
fi

# Kill any existing UI processes
docker-compose exec app sh -c "pkill -f 'run_ui.py' || true" 2>/dev/null || true

# Wait a moment
sleep 2

# Start UI
echo "Starting UI on http://127.0.0.1:7860"
docker-compose exec app python scripts/run_ui.py --api-url http://127.0.0.1:5000

