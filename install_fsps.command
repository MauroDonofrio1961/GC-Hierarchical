#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Run install.command first."
  read -r -p "Press Return to close..."
  exit 1
fi

source .venv/bin/activate
python -m pip install -r requirements-fsps.txt

echo
echo "Python-FSPS installation completed."
echo "The importable package name is 'fsps'."
read -r -p "Press Return to close..."
