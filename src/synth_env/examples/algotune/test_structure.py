"""Test script to verify AlgoTune environment structure without requiring AlgoTune installation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..', 'src'))

try:
    # Test basic imports
    from synth_env.examples.algotune.environment import AlgoTuneEnvironment
    print("✓ Successfully imported AlgoTuneEnvironment")
    
    from synth_env.examples.algotune.environment import _AlgoTuneEngine
    print("✓ Successfully imported _AlgoTuneEngine")
    
    from synth_env.examples.algotune.environment import _AlgoTuneTool
    print("✓ Successfully imported _AlgoTuneTool")
    
    # Check registration
    from synth_env.service.registry import list_supported_env_types
    try:
        supported = list_supported_env_types()
        if "AlgoTune" in supported:
            print("✓ AlgoTune is registered in the environment registry")
        else:
            print("✗ AlgoTune is NOT in the registry (may need to restart service)")
    except Exception as e:
        print(f"✗ Could not check registry: {e}")
    
    print("\nAlgoTune environment structure is correctly set up!")
    print("Note: To run actual tests, install AlgoTune with:")
    print("  pip install -e git+https://github.com/oripress/AlgoTune.git#egg=algotune")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease ensure you're running from the repository root.")