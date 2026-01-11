#!/bin/bash
# Script to use local negmas for development
# This modifies pyproject.toml temporarily - don't commit the changes!

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYPROJECT="$PROJECT_DIR/pyproject.toml"

# Check if local negmas exists
if [ ! -d "$PROJECT_DIR/../negmas" ]; then
    echo "Error: ../negmas directory not found"
    exit 1
fi

# Add the sources section if not present
if ! grep -q '\[tool.uv.sources\]' "$PYPROJECT"; then
    echo "" >> "$PYPROJECT"
    echo "[tool.uv.sources]" >> "$PYPROJECT"
    echo 'negmas = { path = "../negmas", editable = true }' >> "$PYPROJECT"
    echo "Added [tool.uv.sources] section to pyproject.toml"
else
    echo "[tool.uv.sources] section already exists"
fi

# Sync with the new configuration
cd "$PROJECT_DIR"
uv sync

echo ""
echo "Now using local negmas from ../negmas"
echo "IMPORTANT: Don't commit pyproject.toml with the [tool.uv.sources] section!"
