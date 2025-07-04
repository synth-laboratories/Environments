#!/usr/bin/env python3
"""
Test the fixed database queries.
"""
import sys
import pathlib

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from frontend.traces_viewer import database as frontend_db
from backend import database as backend_db
from db_config import db_config

def test_queries():
    """Test all the database queries."""
    print("=== Testing Database Queries ===\n")
    
    # Test 1: Get environments
    print("1. Testing get_environments()...")
    try:
        environments = frontend_db.get_environments()
        print(f"   ✅ Found {len(environments)} environments: {environments}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: List evaluations
    print("\n2. Testing list_evaluations()...")
    try:
        evals = frontend_db.list_evaluations()
        print(f"   ✅ Found {len(evals)} evaluations")
        if len(evals) > 0:
            print(f"   Sample columns: {list(evals.columns)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: List evaluations for specific environment
    print("\n3. Testing list_evaluations(env='crafter')...")
    try:
        evals = frontend_db.list_evaluations(env='crafter')
        print(f"   ✅ Found {len(evals)} evaluations for crafter")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Check backend module
    print("\n4. Testing backend get_environments()...")
    try:
        environments = backend_db.get_environments()
        print(f"   ✅ Backend found {len(environments)} environments")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Sample evaluation data
    print("\n5. Getting sample evaluation data...")
    try:
        con = frontend_db.get_connection()
        
        # Check evaluations table
        eval_count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        print(f"   Total evaluations: {eval_count}")
        
        # Check environments table
        env_count = con.execute("SELECT COUNT(*) FROM environments").fetchone()[0]
        print(f"   Total environments: {env_count}")
        
        # Show environment names
        envs = con.execute("SELECT id, name, display_name FROM environments").fetchall()
        print("   Available environments:")
        for env in envs:
            print(f"     - ID: {env[0]}, Name: {env[1]}, Display: {env[2]}")
        
        con.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Query tests complete!")

if __name__ == "__main__":
    test_queries()