#!/usr/bin/env python3
"""Debug database path resolution from different directories."""
import os
import sys
from pathlib import Path

# Test from different working directories
test_dirs = [
    "/Users/joshuapurtell/Documents/GitHub/Environments",
    "/Users/joshuapurtell/Documents/GitHub/Environments/src/viewer",
    "/Users/joshuapurtell/Documents/GitHub/Environments/src/viewer/frontend",
]

for test_dir in test_dirs:
    print(f"\n=== Working directory: {test_dir} ===")
    os.chdir(test_dir)
    
    # Add viewer to path and import db_config
    viewer_dir = Path("/Users/joshuapurtell/Documents/GitHub/Environments/src/viewer")
    if str(viewer_dir) not in sys.path:
        sys.path.insert(0, str(viewer_dir))
    
    # Import fresh
    if 'db_config' in sys.modules:
        del sys.modules['db_config']
    
    from db_config import db_config
    
    # Re-initialize
    db_config._initialize_db_path()
    
    print(f"Current dir: {os.getcwd()}")
    print(f"DB path: {db_config.db_path}")
    print(f"DB exists: {db_config.validate_db_exists()}")
    
    # Check what's actually at that path
    if db_config.validate_db_exists():
        import duckdb
        con = duckdb.connect(db_config.db_path, read_only=True)
        tables = con.execute("SHOW TABLES").fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
        
        # Check schema of environments table
        env_schema = con.execute("DESCRIBE environments").fetchall()
        print(f"Environments columns: {[c[0] for c in env_schema]}")
        con.close()