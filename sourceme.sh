#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P
)"

PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/.venv"

# Create Python virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip/build tools inside the venv
python3 -m pip install --upgrade pip setuptools wheel

# Install this project in editable mode and upgrade dependencies
python3 -m pip install --upgrade --upgrade-strategy eager -e "$PROJECT_ROOT"

# Optional but useful: make the repo visible even before/without editable install
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Install motulator in editable mode if not already importable
#python -c "import motulator" 2>/dev/null || python3 -m pip install -e "$PROJECT_ROOT"

echo "Motulator development environment ready."
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)" || return 1
