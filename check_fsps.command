#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Run install.command first."
  read -r -p "Press Return to close..."
  exit 1
fi

source .venv/bin/activate
python run.py check --galaxy NGC5128

echo
echo "FSPS is importable in this project environment."
read -r -p "Press Return to close..."
