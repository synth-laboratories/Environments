#!/usr/bin/env python3
"""Debug trace structure to find where images are."""

import json
from pathlib import Path

# Load a trace
trace_file = Path("src/evals/crafter/run_20250703_130153/traces/e8f01cd0-3f58-4ce2-9567-26a02bf6d31e.json")
with open(trace_file, 'r') as f:
    data = json.load(f)

print("=== TRACE STRUCTURE ===")
print(f"Top keys: {list(data.keys())}")

# Look at first turn
if 'trace' in data and 'partition' in data['trace']:
    first_partition = data['trace']['partition'][0]
    print(f"\nFirst partition keys: {list(first_partition.keys())}")
    
    if 'events' in first_partition:
        first_event = first_partition['events'][0]
        print(f"\nFirst event keys: {list(first_event.keys())}")
        
        if 'environment_compute_steps' in first_event:
            env_steps = first_event['environment_compute_steps']
            print(f"\nNumber of env steps: {len(env_steps)}")
            
            if env_steps:
                first_step = env_steps[0]
                print(f"\nFirst env step keys: {list(first_step.keys())}")
                
                # Check compute_output instead of output
                if 'compute_output' in first_step:
                    compute_output = first_step['compute_output']
                    print(f"\nCompute output type: {type(compute_output)}")
                    if isinstance(compute_output, list) and len(compute_output) > 0:
                        output = compute_output[0]
                        print(f"\nFirst compute output keys: {list(output.keys())}")
                    
                    # Check different possible locations for image
                    if 'outputs' in output:
                        print(f"\nOutputs keys: {list(output['outputs'].keys())}")
                        if 'image' in output['outputs']:
                            img = output['outputs']['image']
                            print(f"\nImage data type: {type(img)}")
                            print(f"Image data preview: {str(img)[:100]}...")
                    
                    if 'image' in output:
                        print(f"\nDirect image found!")
                        print(f"Image type: {type(output['image'])}")
                    
                    # Print full output structure
                    print("\n=== FULL OUTPUT STRUCTURE ===")
                    print(json.dumps(output, indent=2)[:1000])
                    print("...")