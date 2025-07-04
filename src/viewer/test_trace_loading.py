#!/usr/bin/env python3
"""
Test trace loading functionality.
"""
import sys
import pathlib

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from frontend.traces_viewer import database as db

def test_trace_loading():
    """Test loading traces."""
    print("=== Testing Trace Loading ===\n")
    
    # Get a sample trace ID
    con = db.get_connection()
    sample = con.execute("""
        SELECT tr.trace_id, tr.model_name, e.run_id 
        FROM trajectories tr
        JOIN evaluations e ON tr.eval_id = e.id
        LIMIT 5
    """).fetchall()
    con.close()
    
    print("Sample traces:")
    for trace_id, model, run_id in sample:
        print(f"  - {trace_id} (model: {model}, run: {run_id})")
    
    if sample:
        # Test loading the first trace
        test_trace_id = sample[0][0]
        print(f"\nTesting get_trace('{test_trace_id}')...")
        
        try:
            trace_data = db.get_trace(test_trace_id)
            if trace_data:
                print("✅ Trace loaded successfully!")
                print(f"   Keys: {list(trace_data.keys())[:10]}...")  # First 10 keys
                print(f"   Model: {trace_data.get('model_name')}")
                print(f"   Success: {trace_data.get('success')}")
                print(f"   Steps: {trace_data.get('num_steps')}")
                if 'trace' in trace_data:
                    print(f"   Has trace data: Yes")
            else:
                print("❌ No trace found")
        except Exception as e:
            print(f"❌ Error loading trace: {e}")

if __name__ == "__main__":
    test_trace_loading()