#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python run.py science --galaxy NGC5128
read -r -p "Press Return to close..."
