#!/usr/bin/env python3
"""
Utility script to verify database path configuration.
Run this to check your database setup.
"""
import os
import sys
import pathlib

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config


def main():
    """Verify database configuration."""
    print("=== Database Path Configuration ===")
    print()
    
    # Get database info
    info = db_config.get_db_info()
    
    print(f"Environment variable TRACE_DB: {info['env_var'] or '(not set)'}")
    print(f"Resolved database path: {info['path']}")
    print(f"Absolute path: {info['absolute_path']}")
    print(f"Parent directory: {info['parent_dir']}")
    print(f"Database exists: {info['exists']}")
    print()
    
    # Check if path is absolute
    if info['path']:
        is_absolute = os.path.isabs(info['path'])
        print(f"Path is absolute: {is_absolute}")
        if not is_absolute:
            print("WARNING: Path is not absolute! This may cause issues.")
    print()
    
    # Suggest fixes if database doesn't exist
    if not info['exists']:
        print("⚠️  Database file not found!")
        print()
        print("To fix this issue, you can:")
        print("1. Set the TRACE_DB environment variable to the correct path:")
        print(f"   export TRACE_DB=/path/to/your/synth_eval.duckdb")
        print()
        print("2. Or ensure the database exists at one of these locations:")
        
        # Show expected locations
        project_root = pathlib.Path(info['absolute_path']).parent if info['absolute_path'] else current_dir.parent.parent
        expected_paths = [
            project_root / "synth_eval.duckdb",
            current_dir / "synth_eval.duckdb",
            project_root / "synth_eval_viewer.duckdb",
        ]
        
        for path in expected_paths:
            exists = "✓" if path.exists() else "✗"
            print(f"   {exists} {path}")
    else:
        print("✅ Database configuration is valid!")
    
    print()
    print("=== Testing Database Connection ===")
    
    try:
        # Try importing and using the database modules
        from frontend.traces_viewer import database as frontend_db
        from backend import database as backend_db
        
        # Try to get a connection
        print("Testing frontend database connection...")
        con = frontend_db.get_connection()
        con.close()
        print("✅ Frontend connection successful!")
        
        print("Testing backend database connection...")
        con = backend_db.get_connection()
        con.close()
        print("✅ Backend connection successful!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Please check the error message above for details.")


if __name__ == "__main__":
    main()