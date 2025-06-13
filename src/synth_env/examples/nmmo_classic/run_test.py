#!/usr/bin/env python3
"""
Runner script for nmmo_classic tests with isolated environment.
This script ensures the nmmo-specific dependencies are available.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Change working directory to nmmo_classic to use local pyproject.toml
nmmo_dir = Path(__file__).parent
os.chdir(nmmo_dir)

# Now import and run the test
try:
    from agent_demos.test_synth_react import test_react_agent_nmmo, eval_react_nmmo

    print("Running NMMO Classic test...")

    # Run the basic test
    test_react_agent_nmmo()
    print("✓ Basic test passed")

    # Run evaluation if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        eval_react_nmmo()
        print("✓ Evaluation completed")

except ImportError as e:
    print(f"Import error: {e}")
    print("Installing nmmo dependencies...")

    # Try to install dependencies using the local pyproject.toml
    import subprocess

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("Dependencies installed. Rerun the script.")
    except subprocess.CalledProcessError as install_error:
        print(f"Failed to install dependencies: {install_error}")
        sys.exit(1)

if __name__ == "__main__":
    pass
