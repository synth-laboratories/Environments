#!/usr/bin/env python3
"""Quick check of database contents."""

from src.synth_env.db import SynthEvalDB

db = SynthEvalDB()
with db.connection() as con:
    # Check tables
    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables:", [t[0] for t in tables])
    
    # Check environments
    envs = con.execute("SELECT * FROM environments").fetchall()
    print(f"\nEnvironments ({len(envs)}):")
    for env in envs:
        print(f"  - {env}")
    
    # Check evaluations
    evals = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    print(f"\nEvaluations: {evals}")
    
    # Check trajectories
    trajs = con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    print(f"Trajectories: {trajs}")
    
    # Check traces
    traces = con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    print(f"Traces: {traces}")