#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Run install.command first."
  read -r -p "Press Return to close..."
  exit 1
fi

source .venv/bin/activate
python -m pytest -q

echo
echo "Step 3 object-model tests completed."
read -r -p "Press Return to close..."
