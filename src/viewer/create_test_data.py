#!/usr/bin/env python3
"""
Create test data for the synth_eval database with proper schema.
"""
import sys
import pathlib
import duckdb
from datetime import datetime, timedelta
import json
import random

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config
from db_schema import assert_valid_schema

def create_test_data():
    """Create comprehensive test data."""
    print("=== Creating Test Data ===\n")
    
    # Validate schema first
    try:
        assert_valid_schema()
        print("✅ Schema validation passed\n")
    except AssertionError as e:
        print(f"❌ Schema validation failed: {e}")
        return
    
    # Get a writable connection
    con = duckdb.connect(db_config.db_path, read_only=False)
    
    try:
        # 1. Insert environments
        print("1. Creating environments...")
        con.execute("""
        INSERT INTO environments (id, name, display_name, description, created_at) 
        VALUES 
            (1, 'crafter', 'Crafter Classic', 'A 2D survival game environment', CURRENT_TIMESTAMP),
            (2, 'minigrid', 'MiniGrid', 'Grid-based reinforcement learning environments', CURRENT_TIMESTAMP),
            (3, 'nethack', 'NetHack', 'Classic roguelike dungeon exploration game', CURRENT_TIMESTAMP),
            (4, 'sokoban', 'Sokoban', 'Japanese puzzle game about pushing boxes', CURRENT_TIMESTAMP)
        """)
        print("   ✅ Created 4 environments")
        
        # 2. Insert evaluations with proper data
        print("\n2. Creating evaluations...")
        base_time = datetime.now() - timedelta(days=7)
        
        eval_data = [
            # Crafter evaluations
            (1, 1, 'crafter-eval-001', base_time, ['gpt-4', 'claude-3'], ['easy', 'medium'], 20, 0.75, 4.2),
            (2, 1, 'crafter-eval-002', base_time + timedelta(days=1), ['gpt-4'], ['hard'], 10, 0.40, 2.1),
            (3, 1, 'crafter-eval-003', base_time + timedelta(days=2), ['claude-3'], ['easy'], 15, 0.87, 5.3),
            
            # MiniGrid evaluations
            (4, 2, 'minigrid-eval-001', base_time + timedelta(hours=12), ['gpt-4', 'claude-3'], ['easy'], 30, 0.93, 8.7),
            (5, 2, 'minigrid-eval-002', base_time + timedelta(days=3), ['gpt-3.5'], ['medium', 'hard'], 25, 0.56, 4.8),
            
            # NetHack evaluations
            (6, 3, 'nethack-eval-001', base_time + timedelta(days=4), ['claude-3'], ['medium'], 12, 0.33, 1.2),
            (7, 3, 'nethack-eval-002', base_time + timedelta(days=5), ['gpt-4'], ['easy', 'medium'], 18, 0.61, 2.8),
            
            # Sokoban evaluations
            (8, 4, 'sokoban-eval-001', base_time + timedelta(days=6), ['gpt-4', 'claude-3'], ['easy', 'medium', 'hard'], 40, 0.70, 15.5),
        ]
        
        for eval_id, env_id, run_id, timestamp, models, difficulties, num_traj, success_rate, avg_achievements in eval_data:
            con.execute("""
            INSERT INTO evaluations (
                id, env_id, run_id, timestamp, models_evaluated, 
                difficulties_evaluated, num_trajectories, 
                success_rate, avg_achievements, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                eval_id, env_id, run_id, timestamp, models, difficulties, 
                num_traj, success_rate, avg_achievements, 
                json.dumps({"test": True, "version": "1.0"})
            ])
        
        print(f"   ✅ Created {len(eval_data)} evaluations")
        
        # 3. Insert some trajectories
        print("\n3. Creating trajectories...")
        traj_id = 1
        for eval_id, env_id, run_id, _, models, difficulties, num_traj, _, _ in eval_data:
            # Create a few trajectories for each evaluation
            for i in range(min(3, num_traj)):  # Max 3 trajectories per eval for test
                model = random.choice(models)
                difficulty = random.choice(difficulties)
                success = random.random() > 0.5
                
                con.execute("""
                INSERT INTO trajectories (
                    id, eval_id, trace_id, model_name, difficulty, 
                    seed, success, final_reward, num_steps, 
                    achievements, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    traj_id, eval_id, f"trace-{run_id}-{i:03d}", model, difficulty,
                    random.randint(1000, 9999), success, 
                    random.uniform(-10, 100) if success else random.uniform(-50, 0),
                    random.randint(50, 500),
                    [f"achievement_{j}" for j in range(random.randint(0, 5))],
                    json.dumps({"test": True})
                ])
                
                # Create corresponding trace
                con.execute("""
                INSERT INTO traces (
                    id, trajectory_id, parquet_path, trace_format, 
                    size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    traj_id, traj_id, 
                    f"traces/{run_id}/trace-{i:03d}.parquet",
                    "parquet", random.randint(1000, 50000)
                ])
                
                traj_id += 1
        
        print(f"   ✅ Created {traj_id - 1} trajectories with traces")
        
        # Commit all changes
        con.commit()
        
        # 4. Verify data
        print("\n4. Verifying data...")
        env_count = con.execute("SELECT COUNT(*) FROM environments").fetchone()[0]
        eval_count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        traj_count = con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        trace_count = con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        
        print(f"   Environments: {env_count}")
        print(f"   Evaluations: {eval_count}")
        print(f"   Trajectories: {traj_count}")
        print(f"   Traces: {trace_count}")
        
        print("\n✅ Test data created successfully!")
        
    except Exception as e:
        print(f"\n❌ Error creating data: {e}")
        con.rollback()
        raise
    finally:
        con.close()

def clear_all_data():
    """Clear all data from the database."""
    print("=== Clearing All Data ===\n")
    
    con = duckdb.connect(db_config.db_path, read_only=False)
    
    try:
        # Delete in correct order due to foreign keys
        tables = ['traces', 'trajectories', 'evaluations', 'environments']
        for table in tables:
            con.execute(f"DELETE FROM {table}")
            print(f"   Cleared {table}")
        
        con.commit()
        print("\n✅ All data cleared!")
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        con.rollback()
    finally:
        con.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', help='Clear all data')
    args = parser.parse_args()
    
    if args.clear:
        clear_all_data()
    else:
        # Clear then create fresh data
        clear_all_data()
        print()
        create_test_data()