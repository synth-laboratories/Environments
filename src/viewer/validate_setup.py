#!/usr/bin/env python3
"""
Validate the complete viewer setup.
"""
import sys
import pathlib

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config
from db_schema import assert_valid_schema, SCHEMA
from frontend.traces_viewer import database as db

def validate_setup():
    """Run complete validation of the viewer setup."""
    print("=== Validating Viewer Setup ===\n")
    
    all_good = True
    
    # 1. Check database path
    print("1. Database Configuration:")
    print(f"   Path: {db_config.db_path}")
    print(f"   Exists: {db_config.validate_db_exists()}")
    print(f"   Absolute: {pathlib.Path(db_config.db_path).is_absolute()}")
    
    if not db_config.validate_db_exists():
        print("   ❌ Database file not found!")
        all_good = False
    else:
        print("   ✅ Database configuration valid")
    
    # 2. Validate schema
    print("\n2. Schema Validation:")
    try:
        assert_valid_schema()
        print("   ✅ Schema is valid")
        
        # Show expected tables
        print("   Expected tables:")
        for table in SCHEMA:
            if not SCHEMA[table].get("is_view"):
                print(f"     - {table}")
    except AssertionError as e:
        print(f"   ❌ Schema validation failed: {e}")
        all_good = False
    
    # 3. Test queries
    print("\n3. Testing Database Queries:")
    
    try:
        # Test get_environments
        envs = db.get_environments()
        print(f"   ✅ get_environments() returned {len(envs)} environments")
        
        # Test list_evaluations
        evals = db.list_evaluations()
        print(f"   ✅ list_evaluations() returned {len(evals)} evaluations")
        
        # Test with specific environment if any exist
        if envs:
            env_evals = db.list_evaluations(env=envs[0])
            print(f"   ✅ list_evaluations(env='{envs[0]}') returned {len(env_evals)} evaluations")
    except Exception as e:
        print(f"   ❌ Query error: {e}")
        all_good = False
    
    # 4. Check data
    print("\n4. Database Contents:")
    con = db.get_connection()
    try:
        for table in ["environments", "evaluations", "trajectories", "traces"]:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"   {table}: {count} rows")
    except Exception as e:
        print(f"   ❌ Error checking contents: {e}")
        all_good = False
    finally:
        con.close()
    
    # Summary
    print("\n" + "="*50)
    if all_good:
        print("✅ All validations passed! The viewer is ready to use.")
        print("\nTo run the viewer:")
        print("  cd frontend && reflex run")
        print("\nOr use the helper script:")
        print("  ./run_viewer.sh")
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(validate_setup())