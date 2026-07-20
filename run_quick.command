#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python run.py quick --galaxy NGC5128
read -r -p "Press Return to close..."
