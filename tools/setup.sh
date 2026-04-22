#!/bin/bash
# setup.sh — Linux-only bootstrap (calls the cross-platform Python script)
# Usage:  bash setup.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing Linux system packages ==="
sudo apt update
sudo apt install -y python3-pip python3-venv libiio0 libiio-dev libiio-utils \
                    python3-pyqt5 python3-pyqtgraph

echo ""
echo "=== Running cross-platform setup ==="
python3 "$SCRIPT_DIR/setup.py"
