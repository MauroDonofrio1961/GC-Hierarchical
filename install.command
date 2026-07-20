#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "GC-Hierarchical 1.0-alpha — Step 1 installation"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: Python 3 was not found."
  echo "Install Python 3.9 or newer, then run this file again."
  read -r -p "Press Return to close..."
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_OK=$(python3 -c 'import sys; print(int(sys.version_info >= (3, 9)))')

echo "Using Python ${PYTHON_VERSION}"

if [ "$PYTHON_OK" != "1" ]; then
  echo "ERROR: Python 3.9 or newer is required."
  read -r -p "Press Return to close..."
  exit 1
fi

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Installation completed successfully."
echo "FSPS is not required for Step 1 and was not installed."
echo "Next, double-click run.command."
read -r -p "Press Return to close..."
