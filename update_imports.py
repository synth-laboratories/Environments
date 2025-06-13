#!/usr/bin/env python3
import re
import os

def update_imports(file_path):
    """Update imports in a Python file to use synth_env prefix."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match imports
    patterns = [
        (r'^(from\s+)(environment|service|stateful|tasks|examples|reproducibility|v0_observability)(\.|$)', r'\1synth_env.\2\3'),
        (r'^(import\s+)(environment|service|stateful|tasks|examples|reproducibility|v0_observability)(\.|$)', r'\1synth_env.\2\3'),
    ]
    
    lines = content.split('\n')
    updated_lines = []
    changed = False
    
    for line in lines:
        new_line = line
        for pattern, replacement in patterns:
            if re.match(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    changed = True
                    print(f"  {line} -> {new_line}")
                break
        updated_lines.append(new_line)
    
    if changed:
        with open(file_path, 'w') as f:
            f.write('\n'.join(updated_lines))
        return True
    return False

# List of files to update
files_to_update = [
    "src/synth_env/examples/crafter_classic/environment.py",
    "src/synth_env/examples/enron/agent_demos/test_synth_react.py",
    "src/synth_env/examples/enron/art_helpers/email_search_tools.py",
    "src/synth_env/examples/enron/engine.py",
    "src/synth_env/examples/enron/environment.py",
    "src/synth_env/examples/enron/units/keyword_stats.py",
    "src/synth_env/examples/enron/units/test_email_index.py",
    "src/synth_env/examples/math/agent_demos/plan_execute.py",
    "src/synth_env/examples/math/environment.py",
    "src/synth_env/examples/math/taskset.py",
    "src/synth_env/examples/sokoban/agent_demos/test_synth_react_locally.py",
    "src/synth_env/examples/sokoban/agent_demos/test_synth_react_service.py",
    "src/synth_env/examples/sokoban/engine.py",
    "src/synth_env/examples/sokoban/taskset.py",
    "src/synth_env/examples/sokoban/units/test_building_task_set.py",
    "src/synth_env/examples/sokoban/units/test_false_positive.py",
    "src/synth_env/examples/sokoban/units/test_simple_run_through_environment.py",
    "src/synth_env/examples/sokoban/units/test_sokoban_environment.py",
    "src/synth_env/examples/sokoban/units/test_tree.py",
    "tests/integration/test_service_api.py",
    "tests/integration/test_sokoban_service.py",
    "tests/unit/test_external_registry.py",
    "tests/unit/test_math_environment.py",
    "tests/unit/test_sokoban_qstar.py"
]

print("Updating imports in Python files...")
updated_count = 0

for file_path in files_to_update:
    if os.path.exists(file_path):
        print(f"\nProcessing {file_path}:")
        if update_imports(file_path):
            updated_count += 1
    else:
        print(f"File not found: {file_path}")

print(f"\n\nTotal files updated: {updated_count}")