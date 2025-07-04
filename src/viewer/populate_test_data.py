#!/usr/bin/env python3
"""
Populate test data in the database for testing.
WARNING: This modifies the database! Only use for testing.
"""
import sys
import pathlib
import duckdb
from datetime import datetime

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config

def populate_test_data():
    """Populate test data in the database."""
    print("=== Populating Test Data ===\n")
    
    # Get a writable connection
    con = duckdb.connect(db_config.db_path, read_only=False)
    
    try:
        # Insert test environments
        print("1. Inserting test environments...")
        # Check if environments already exist
        existing = con.execute("SELECT name FROM environments WHERE name IN ('crafter', 'minigrid', 'nethack')").fetchall()
        if existing:
            print(f"   ⚠️  Some environments already exist: {[e[0] for e in existing]}")
        else:
            con.execute("""
            INSERT INTO environments (id, name, display_name, description, created_at) 
            VALUES 
                (1, 'crafter', 'Crafter Classic', 'A 2D survival game environment', CURRENT_TIMESTAMP),
                (2, 'minigrid', 'MiniGrid', 'Grid-based RL environments', CURRENT_TIMESTAMP),
                (3, 'nethack', 'NetHack', 'Classic roguelike game', CURRENT_TIMESTAMP)
            """)
        print("   ✅ Inserted 3 test environments")
        
        # Get environment IDs
        env_ids = con.execute("SELECT id, name FROM environments").fetchall()
        env_map = {name: id for id, name in env_ids}
        
        # Insert test evaluations
        print("\n2. Inserting test evaluations...")
        con.execute(f"""
        INSERT INTO evaluations (
            env_id, run_id, timestamp, models_evaluated, 
            difficulties_evaluated, num_trajectories, 
            success_rate, avg_achievements, metadata, created_at
        ) VALUES 
            ({env_map['crafter']}, 'test-run-001', CURRENT_TIMESTAMP, 
             ['gpt-4', 'claude-3'], ['easy', 'medium'], 10, 
             0.75, 3.5, '{{"test": true}}', CURRENT_TIMESTAMP),
            ({env_map['minigrid']}, 'test-run-002', CURRENT_TIMESTAMP, 
             ['gpt-4'], ['easy'], 5, 
             0.80, 2.0, '{{"test": true}}', CURRENT_TIMESTAMP)
        """)
        print("   ✅ Inserted 2 test evaluations")
        
        # Verify data
        print("\n3. Verifying data...")
        env_count = con.execute("SELECT COUNT(*) FROM environments").fetchone()[0]
        eval_count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        print(f"   Environments: {env_count}")
        print(f"   Evaluations: {eval_count}")
        
        con.commit()
        print("\n✅ Test data populated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error populating data: {e}")
        try:
            con.rollback()
        except:
            pass  # Rollback might fail if no transaction
    finally:
        con.close()

def clear_test_data():
    """Clear test data from the database."""
    print("\n=== Clearing Test Data ===\n")
    
    con = duckdb.connect(db_config.db_path, read_only=False)
    
    try:
        # Delete in correct order due to foreign keys
        con.execute("DELETE FROM trajectories")
        con.execute("DELETE FROM traces")
        con.execute("DELETE FROM evaluations")
        con.execute("DELETE FROM environments WHERE name IN ('crafter', 'minigrid', 'nethack')")
        con.commit()
        print("✅ Test data cleared!")
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        con.rollback()
    finally:
        con.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', help='Clear test data instead of populating')
    args = parser.parse_args()
    
    if args.clear:
        clear_test_data()
    else:
        populate_test_data()