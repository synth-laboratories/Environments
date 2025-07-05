#!/usr/bin/env python3
"""
Definitive database schema definition for synth_eval.
This is the SINGLE SOURCE OF TRUTH for the database schema.
"""

import duckdb
from typing import Dict, List, Tuple
import sys
import pathlib

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import db_config

# DEFINITIVE SCHEMA DEFINITION
SCHEMA = {
    "environments": {
        "columns": [
            ("id", "INTEGER PRIMARY KEY"),
            ("name", "VARCHAR NOT NULL UNIQUE"),
            ("display_name", "VARCHAR NOT NULL"),
            ("description", "VARCHAR"),
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ],
        "indexes": ["CREATE INDEX idx_environments_name ON environments(name)"],
    },
    "evaluations": {
        "columns": [
            ("id", "INTEGER PRIMARY KEY"),
            ("env_id", "INTEGER NOT NULL REFERENCES environments(id)"),
            ("run_id", "VARCHAR NOT NULL UNIQUE"),
            ("timestamp", "TIMESTAMP NOT NULL"),
            ("models_evaluated", "VARCHAR[] NOT NULL"),
            ("difficulties_evaluated", "VARCHAR[]"),
            ("num_trajectories", "INTEGER NOT NULL"),
            ("success_rate", "FLOAT"),
            ("avg_achievements", "FLOAT"),
            ("metadata", "JSON"),
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ],
        "indexes": [
            "CREATE INDEX idx_evaluations_env_id ON evaluations(env_id)",
            "CREATE INDEX idx_evaluations_run_id ON evaluations(run_id)",
            "CREATE INDEX idx_evaluations_timestamp ON evaluations(timestamp)",
        ],
    },
    "trajectories": {
        "columns": [
            ("id", "INTEGER PRIMARY KEY"),
            ("eval_id", "INTEGER NOT NULL REFERENCES evaluations(id)"),
            ("trace_id", "VARCHAR NOT NULL UNIQUE"),
            ("model_name", "VARCHAR NOT NULL"),
            ("difficulty", "VARCHAR"),
            ("seed", "INTEGER"),
            ("success", "BOOLEAN NOT NULL"),
            ("final_reward", "FLOAT"),
            ("num_steps", "INTEGER NOT NULL"),
            ("achievements", "VARCHAR[]"),
            ("metadata", "JSON"),
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ],
        "indexes": [
            "CREATE INDEX idx_trajectories_eval_id ON trajectories(eval_id)",
            "CREATE INDEX idx_trajectories_trace_id ON trajectories(trace_id)",
        ],
    },
    "traces": {
        "columns": [
            ("id", "INTEGER PRIMARY KEY"),
            ("trajectory_id", "INTEGER NOT NULL REFERENCES trajectories(id)"),
            ("parquet_path", "VARCHAR"),
            ("trace_format", "VARCHAR DEFAULT 'parquet'"),
            ("size_bytes", "INTEGER"),
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ],
        "indexes": ["CREATE INDEX idx_traces_trajectory_id ON traces(trajectory_id)"],
    },
    "evaluation_summary": {
        "is_view": True,
        "definition": """
        CREATE VIEW evaluation_summary AS
        SELECT 
            e.id as eval_id,
            env.name as env_name,
            env.display_name as env_display_name,
            e.run_id,
            e.timestamp,
            e.models_evaluated,
            e.num_trajectories,
            e.success_rate
        FROM evaluations e
        JOIN environments env ON e.env_id = env.id
        """,
    },
}


def create_schema(con: duckdb.DuckDBPyConnection, drop_existing: bool = False):
    """Create the database schema."""
    if drop_existing:
        # Drop in reverse order due to foreign keys
        tables = [
            "evaluation_summary",
            "traces",
            "trajectories",
            "evaluations",
            "environments",
        ]
        for table in tables:
            try:
                if table == "evaluation_summary":
                    con.execute(f"DROP VIEW IF EXISTS {table}")
                else:
                    con.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"   Dropped {table}")
            except Exception as e:
                print(f"   Warning dropping {table}: {e}")

    # Create tables in order
    for table_name, table_def in SCHEMA.items():
        if table_def.get("is_view"):
            # Create view
            con.execute(table_def["definition"])
            print(f"   Created view: {table_name}")
        else:
            # Build CREATE TABLE statement
            columns = ", ".join(
                [f"{col} {dtype}" for col, dtype in table_def["columns"]]
            )
            create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
            con.execute(create_stmt)
            print(f"   Created table: {table_name}")

            # Create indexes
            for index_stmt in table_def.get("indexes", []):
                con.execute(index_stmt)


def validate_schema(con: duckdb.DuckDBPyConnection) -> Tuple[bool, List[str]]:
    """Validate that the database matches the expected schema."""
    errors = []

    # Check all tables exist
    existing_tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
    expected_tables = {name for name, def_ in SCHEMA.items() if not def_.get("is_view")}

    missing_tables = expected_tables - existing_tables
    if missing_tables:
        errors.append(f"Missing tables: {missing_tables}")

    # Check each table's schema
    for table_name, table_def in SCHEMA.items():
        if table_def.get("is_view") or table_name not in existing_tables:
            continue

        # Get actual columns
        actual_cols = con.execute(f"DESCRIBE {table_name}").fetchall()
        actual_col_names = {col[0] for col in actual_cols}

        # Get expected columns
        expected_col_names = {col[0] for col in table_def["columns"]}

        # Check for missing/extra columns
        missing_cols = expected_col_names - actual_col_names
        if missing_cols:
            errors.append(f"Table {table_name} missing columns: {missing_cols}")

        extra_cols = actual_col_names - expected_col_names
        if extra_cols:
            errors.append(f"Table {table_name} has extra columns: {extra_cols}")

    return len(errors) == 0, errors


def assert_valid_schema():
    """Assert that the database has the correct schema."""
    con = duckdb.connect(db_config.db_path, read_only=True)
    try:
        valid, errors = validate_schema(con)
        if not valid:
            raise AssertionError(
                f"Database schema validation failed:\n" + "\n".join(errors)
            )
    finally:
        con.close()


def recreate_database():
    """Recreate the database with the correct schema."""
    print("=== Recreating Database with Correct Schema ===\n")

    con = duckdb.connect(db_config.db_path, read_only=False)
    try:
        # Create schema
        print("1. Creating schema...")
        create_schema(con, drop_existing=True)

        # Validate
        print("\n2. Validating schema...")
        valid, errors = validate_schema(con)
        if valid:
            print("   ✅ Schema is valid!")
        else:
            print("   ❌ Schema validation failed:")
            for error in errors:
                print(f"      - {error}")
            raise Exception("Schema validation failed")

        con.commit()
        print("\n✅ Database recreated successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate", action="store_true", help="Validate current schema"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate database with correct schema"
    )
    args = parser.parse_args()

    if args.validate:
        try:
            assert_valid_schema()
            print("✅ Schema is valid!")
        except AssertionError as e:
            print(f"❌ {e}")
            sys.exit(1)
    elif args.recreate:
        recreate_database()
    else:
        # Show current schema
        con = duckdb.connect(db_config.db_path, read_only=True)
        print("=== Current Database Schema ===\n")
        tables = con.execute("SHOW TABLES").fetchall()
        for table in tables:
            print(f"{table[0]}:")
            schema = con.execute(f"DESCRIBE {table[0]}").fetchall()
            for col in schema:
                print(f"  - {col[0]}: {col[1]}")
            print()
        con.close()
