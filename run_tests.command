#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python -m pytest -q
read -r -p "Press Return to close..."
