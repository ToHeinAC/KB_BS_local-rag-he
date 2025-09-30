#!/bin/bash
# Launcher script for the RAG application
# Usage: ./run.sh [app_version]
# Examples:
#   ./run.sh          # Runs default app (app_v2_0.py)
#   ./run.sh v2_0     # Runs app_v2_0.py
#   ./run.sh v2_0g    # Runs app_v2_0g.py (German)
#   ./run.sh v1_1     # Runs app_v1_1.py

# Set default app version
APP_VERSION="${1:-v2_0}"

# Map version to app file
case "$APP_VERSION" in
    v2_0)
        APP_FILE="apps/app_v2_0.py"
        ;;
    v2_0g)
        APP_FILE="apps/app_v2_0g.py"
        ;;
    v1_1)
        APP_FILE="apps/app_v1_1.py"
        ;;
    *)
        echo "Unknown version: $APP_VERSION"
        echo "Available versions: v2_0, v2_0g, v1_1"
        exit 1
        ;;
esac

# Check if file exists
if [ ! -f "$APP_FILE" ]; then
    echo "Error: App file not found: $APP_FILE"
    exit 1
fi

echo "🚀 Starting RAG Application ($APP_VERSION)..."
echo "📂 App file: $APP_FILE"
echo ""

# Run with uv
uv run streamlit run "$APP_FILE" --server.port 8501 --server.address localhost
