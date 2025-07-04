#!/usr/bin/env python3
"""
Inspect database schema to find correct column names.
"""
import duckdb
import sys
import pathlib

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config

def inspect_schema():
    """Inspect the database schema."""
    print("=== Database Schema Inspection ===\n")
    
    try:
        # Get connection
        con = duckdb.connect(db_config.db_path, read_only=True)
        
        # List all tables
        tables = con.execute("SHOW TABLES").fetchall()
        print("Tables in database:")
        for table in tables:
            print(f"  - {table[0]}")
        print()
        
        # Check if evaluations table exists
        if any(t[0] == 'evaluations' for t in tables):
            print("Schema for 'evaluations' table:")
            schema = con.execute("DESCRIBE evaluations").fetchall()
            for col in schema:
                print(f"  - {col[0]}: {col[1]}")
            print()
            
            # Show sample data
            print("Sample data from evaluations (first 5 rows):")
            sample = con.execute("SELECT * FROM evaluations LIMIT 5").fetchdf()
            print(sample)
            print()
        
        # Check for other relevant tables
        for table_name in ['traces', 'trajectories', 'runs', 'environments']:
            if any(t[0] == table_name for t in tables):
                print(f"\nSchema for '{table_name}' table:")
                schema = con.execute(f"DESCRIBE {table_name}").fetchall()
                for col in schema:
                    print(f"  - {col[0]}: {col[1]}")
        
        con.close()
        
    except Exception as e:
        print(f"Error inspecting database: {e}")
        print("\nTrying to get more details...")
        
        # Try a simpler query
        try:
            con = duckdb.connect(db_config.db_path, read_only=True)
            # Just list tables
            print("\nTables available:")
            result = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
            for row in result:
                print(f"  - {row[0]}")
            con.close()
        except Exception as e2:
            print(f"Secondary error: {e2}")

if __name__ == "__main__":
    inspect_schema()