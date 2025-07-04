#!/bin/bash

# AlgoTune environment evaluation runner
# Usage: ./run_env_eval.sh [config_file]

# Default to algotune_quick.toml if no config specified
CONFIG=${1:-"evals/configs/algotune_quick.toml"}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the root of the repository
cd "$SCRIPT_DIR/../../../.." || exit 1

# Run the evaluation
python src/run_eval.py --config "$CONFIG"