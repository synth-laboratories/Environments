#!/usr/bin/env python3
"""
Test the DuckDB-backed viewer by importing some Crafter evaluations and viewing them.
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

from src.synth_env.viewer.duckdb_integrated_server import run_duckdb_integrated_server
from src.synth_env.db import SynthEvalDB
import subprocess
import time


def setup_test_data():
    """Import some Crafter evaluations into DuckDB for testing."""
    print("🔄 Setting up test data...")
    
    # Check if we have any Crafter evaluations
    eval_dir = Path("src/evals/crafter")
    if not eval_dir.exists():
        print("❌ No Crafter evaluations found at src/evals/crafter")
        print("   Please run some evaluations first")
        return False
    
    runs = list(eval_dir.glob("run_*"))
    if not runs:
        print("❌ No evaluation runs found")
        return False
    
    print(f"✅ Found {len(runs)} Crafter evaluation runs")
    
    # Import them to DuckDB
    print("\n📥 Importing evaluations to DuckDB...")
    
    # Run import script with limit
    cmd = [
        sys.executable, 
        "scripts/import_eval_dirs_to_db.py",
        "--db-path", "synth_eval_test.duckdb",
        "--env", "crafter",
        "--limit", "3",  # Just import first 3 for testing
        "--no-parquet"  # Skip parquet for now to speed up testing
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Import failed:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    
    # Check database
    db = SynthEvalDB("synth_eval_test.duckdb")
    with db.connection() as con:
        eval_count = con.execute("""
            SELECT COUNT(*) FROM evaluations e
            JOIN environments env ON e.env_id = env.id
            WHERE env.name = 'crafter'
        """).fetchone()[0]
        
        traj_count = con.execute("""
            SELECT COUNT(*) FROM trajectories t
            JOIN evaluations e ON t.eval_id = e.id
            JOIN environments env ON e.env_id = env.id
            WHERE env.name = 'crafter'
        """).fetchone()[0]
    
    print(f"\n✅ Database contains:")
    print(f"   - {eval_count} Crafter evaluations")
    print(f"   - {traj_count} trajectories")
    
    return True


async def main():
    """Main test function."""
    print("🧪 Testing DuckDB-backed Synth Viewer")
    print("=" * 50)
    
    # Setup test data
    if not setup_test_data():
        print("\n❌ Failed to setup test data")
        return
    
    # Start server
    print("\n🚀 Starting DuckDB-backed viewer on port 8997...")
    print("\n📌 To test:")
    print("   1. Navigate to http://localhost:8997")
    print("   2. You should see the landing page with '(DuckDB)' indicator")
    print("   3. Click on Crafter to see evaluations from the database")
    print("   4. Click 'View' on any evaluation to test the viewer")
    print("\n   The URL should show: /viewer?eval_dir=db://crafter/run_xxx&storage=duckdb")
    print("\n   Press Ctrl+C to stop\n")
    
    # Run server
    await run_duckdb_integrated_server(
        port=8997,
        db_path="synth_eval_test.duckdb",
        base_dir=Path("src/evals"),
        parquet_dir=Path("src/evals_parquet")
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test stopped")