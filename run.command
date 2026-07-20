#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "The environment is not installed."
  echo "Run install.command first."
  read -r -p "Press Return to close..."
  exit 1
fi

source .venv/bin/activate
python run.py prepare --galaxy NGC5128

echo
echo "Preparation finished successfully."
echo "Upload the file data/NGC5128/processed/manifest.json"
echo "or the complete results/NGC5128 folder for review."
read -r -p "Press Return to close..."
