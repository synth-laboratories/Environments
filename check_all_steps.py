#!/usr/bin/env python3
"""Check all environment steps for images."""

import json
from pathlib import Path

trace_file = Path("src/evals/crafter/run_20250703_130153/traces/e8f01cd0-3f58-4ce2-9567-26a02bf6d31e.json")
with open(trace_file, 'r') as f:
    data = json.load(f)

found_image = False

# Check all partitions
for p_idx, partition in enumerate(data['trace']['partition']):
    for e_idx, event in enumerate(partition['events']):
        for s_idx, step in enumerate(event.get('environment_compute_steps', [])):
            if 'compute_output' in step and step['compute_output']:
                output = step['compute_output'][0]
                outputs = output.get('outputs', {})
                
                # Check for image in various places
                if 'image' in outputs:
                    print(f"Found image in partition {p_idx}, event {e_idx}, step {s_idx}")
                    print(f"Image data: {str(outputs['image'])[:100]}...")
                    found_image = True
                    break
                
                # Check step_type
                if 'step_type' in step:
                    print(f"Step type at {p_idx},{e_idx},{s_idx}: {step['step_type']}")
                
                # Check for observation
                if 'observation' in outputs:
                    print(f"Found observation at {p_idx},{e_idx},{s_idx}")
                    obs = outputs['observation']
                    if isinstance(obs, dict):
                        print(f"  Observation keys: {list(obs.keys())}")
                    else:
                        print(f"  Observation type: {type(obs)}")
    
    if found_image:
        break

if not found_image:
    print("\nNo images found in any environment steps!")
    print("\nChecking step types...")
    
    # Let's see what types of steps we have
    step_types = set()
    for partition in data['trace']['partition']:
        for event in partition['events']:
            for step in event.get('environment_compute_steps', []):
                if 'step_type' in step:
                    step_types.add(step['step_type'])
    
    print(f"Found step types: {step_types}")
    
    # Check the actual structure of a full step
    print("\n=== FULL ENVIRONMENT STEP STRUCTURE ===")
    first_step = data['trace']['partition'][0]['events'][0]['environment_compute_steps'][0]
    print(json.dumps(first_step, indent=2)[:2000])