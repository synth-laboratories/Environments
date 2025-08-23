#!/usr/bin/env python3
"""
Reproduce the synth_env import issue
"""

print("=== Testing synth_env import issue ===\n")

print("1. Attempting to import synth_env:")
try:
    import synth_env
    print("✓ Success: synth_env imported")
except ImportError as e:
    print(f"✗ Failed: {e}")

print("\n2. Checking what's in the synth_env package directory:")
import os
import sys
import site
print(f"   Python path: {sys.path}")

# Try to find where synth_env is installed
for path in sys.path:
    potential_path = os.path.join(path, "synth_env")
    if os.path.exists(potential_path):
        print(f"   Found synth_env at: {potential_path}")
        contents = os.listdir(potential_path)
        print(f"   Contents: {contents}")
        
        print("\n3. Looking at synth_env/__init__.py:")
        init_file = os.path.join(potential_path, "__init__.py")
        if os.path.exists(init_file):
            with open(init_file, 'r') as f:
                print("   " + "\n   ".join(f.read().splitlines()[:10]))  # First 10 lines
        break
else:
    print("   synth_env not found in any Python path")

print("\n4. The issue:")
print("   - synth_env/__init__.py tries to import 'service' module")
print("   - But there's no 'service' directory in the package")
print("   - This causes: ImportError: cannot import name 'service'")

print("\n5. What we need from synth_env for our tests:")
print("   - synth_env.schema (for ToolCall, StepResult, etc.)")
print("   - synth_env.environment (for Environment base classes)")
print("   - But these imports fail due to the 'service' import error")