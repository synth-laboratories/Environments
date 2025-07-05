import streamlit as st
import duckdb
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import os
from datetime import datetime
import glob

# Configure Streamlit page
st.set_page_config(
    page_title="Synth Trace Viewer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Database connection
def get_db_connection():
    """Get DuckDB connection."""
    # Try environment variable first
    db_path_env = os.getenv("TRACE_DB")
    if db_path_env:
        db_path = Path(db_path_env)
        if not db_path.exists():
            st.error(
                f"Database not found at {db_path} (from TRACE_DB environment variable)"
            )
            return None
    else:
        # Try multiple possible locations for the database
        possible_paths = [
            Path(__file__).parent.parent.parent
            / "synth_eval.duckdb",  # Environments/synth_eval.duckdb
            Path.cwd() / "synth_eval.duckdb",  # Current working directory
            Path("synth_eval.duckdb"),  # Relative to current directory
            Path("../synth_eval.duckdb"),  # One level up
            Path("../../synth_eval.duckdb"),  # Two levels up
            Path("../../../synth_eval.duckdb"),  # Three levels up
        ]

        db_path = None
        for path in possible_paths:
            if path.exists():
                db_path = path
                break

        if not db_path:
            st.error(f"Database not found. Tried the following locations:")
            for path in possible_paths:
                st.error(f"  - {path.resolve()} (exists: {path.exists()})")
            st.info(
                "Please ensure TRACE_DB environment variable is set or synth_eval.duckdb exists in one of the expected locations"
            )
            return None

    try:
        conn = duckdb.connect(str(db_path))
        # Ensure archived column exists
        try:
            conn.execute(
                "ALTER TABLE evaluations ADD COLUMN archived BOOLEAN DEFAULT FALSE"
            )
        except:
            # Column already exists or other error - ignore
            pass
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None


# Data loading functions
def load_environments(conn):
    """Load available environments."""
    try:
        result = conn.execute(
            "SELECT id, name, display_name FROM environments ORDER BY name"
        ).fetchall()
        return [
            {"id": row[0], "name": row[1], "display_name": row[2]} for row in result
        ]
    except Exception as e:
        st.error(f"Error loading environments: {e}")
        return []


def load_evaluations(conn, env_id, archived=False):
    """Load evaluations for an environment."""
    try:
        query = """
        SELECT id, env_id, run_id, timestamp, models_evaluated, num_trajectories, success_rate, avg_achievements, 
               COALESCE(archived, FALSE) as archived
        FROM evaluations 
        WHERE env_id = ? AND COALESCE(archived, FALSE) = ?
        ORDER BY timestamp DESC
        """
        result = conn.execute(query, [env_id, archived]).fetchall()
        return [
            {
                "id": row[0],
                "env_id": row[1],
                "run_id": row[2],
                "timestamp": row[3],
                "models_evaluated": row[4],
                "num_trajectories": row[5],
                "success_rate": row[6],
                "avg_achievements": row[7],
                "archived": row[8],
            }
            for row in result
        ]
    except Exception as e:
        st.error(f"Error loading evaluations: {e}")
        return []


def archive_evaluation(conn, evaluation_id, archived=True):
    """Archive or unarchive an evaluation."""
    try:
        conn.execute(
            "UPDATE evaluations SET archived = ? WHERE id = ?",
            [archived, evaluation_id],
        )
        return True
    except Exception as e:
        st.error(f"Error archiving evaluation: {e}")
        return False


def format_timestamp(timestamp):
    """Format timestamp to readable string."""
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except:
            return timestamp
    else:
        dt = timestamp

    return dt.strftime("%m/%d %H:%M")


def has_agent_reasoning(conn, trajectory_id):
    """Check if a trajectory has proper agent compute steps with reasoning."""
    try:
        # Get trace data
        trace_data = load_trace_data(conn, trajectory_id)
        if not trace_data:
            return False

        # Check if trace has agent compute steps
        trace_obj = trace_data.get("trace", {})
        partitions = trace_obj.get("partition", [])

        # Look for agent compute steps in any partition
        for partition in partitions:
            events = partition.get("events", [])
            for event in events:
                if "agent_compute_step" in event:
                    agent_step = event["agent_compute_step"]
                    # Check if it has actual reasoning (messages)
                    # Try both possible structures
                    messages = None
                    if "inputs" in agent_step and "messages" in agent_step["inputs"]:
                        messages = agent_step["inputs"]["messages"]
                    elif (
                        "compute_input" in agent_step
                        and len(agent_step["compute_input"]) > 0
                    ):
                        if "messages" in agent_step["compute_input"][0]:
                            messages = agent_step["compute_input"][0]["messages"]

                    if messages and len(messages) > 0:
                        return True

        return False

    except Exception as e:
        # If we can't check, assume it doesn't have reasoning
        return False


def load_trajectories(conn, evaluation_id):
    """Load trajectories for an evaluation, filtering out incomplete traces."""
    try:
        query = """
        SELECT id, eval_id, trace_id, model_name, difficulty, seed, success, final_reward, num_steps, achievements, metadata
        FROM trajectories 
        WHERE eval_id = ? 
        ORDER BY final_reward DESC
        """
        result = conn.execute(query, [evaluation_id]).fetchall()

        # Filter trajectories to only include those with agent reasoning
        filtered_trajectories = []
        for row in result:
            trajectory = {
                "id": row[0],
                "eval_id": row[1],
                "trace_id": row[2],
                "model_name": row[3],
                "difficulty": row[4],
                "seed": row[5],
                "success": row[6],
                "final_reward": row[7],
                "num_steps": row[8],
                "achievements": json.loads(row[9]) if row[9] else [],
                "metadata": json.loads(row[10]) if row[10] else {},
            }

            # Only include trajectories with proper agent reasoning
            if has_agent_reasoning(conn, trajectory["id"]):
                filtered_trajectories.append(trajectory)

        return filtered_trajectories

    except Exception as e:
        st.error(f"Error loading trajectories: {e}")
        return []


def load_trace_data(conn, trajectory_id):
    """Load full trace data from JSON file."""
    try:
        # Get trace file path from database
        query = """
        SELECT t.parquet_path, traj.trace_id, traj.eval_id
        FROM traces t
        JOIN trajectories traj ON t.trajectory_id = traj.id
        WHERE traj.id = ?
        """
        result = conn.execute(query, [trajectory_id]).fetchone()

        if not result:
            return None

        file_path, trace_id, eval_id = result
        trace_file = Path(file_path)

        # Try the database path first
        if trace_file.exists():
            with open(trace_file, "r") as f:
                return json.load(f)

        # If database path doesn't exist, try to find the trace file using fallback logic
        # Get evaluation info to determine environment and run_id
        eval_query = """
        SELECT e.run_id, env.name as env_name
        FROM evaluations e
        JOIN environments env ON e.env_id = env.id
        WHERE e.id = ?
        """
        eval_result = conn.execute(eval_query, [eval_id]).fetchone()

        if eval_result:
            run_id, env_name = eval_result

            # Try multiple possible locations for the trace file
            possible_paths = [
                # Current structure
                Path(f"../../src/evals/{env_name}/{run_id}/traces/{trace_id}.json"),
                Path(f"../src/evals/{env_name}/{run_id}/traces/{trace_id}.json"),
                Path(f"../../../src/evals/{env_name}/{run_id}/traces/{trace_id}.json"),
                # Legacy structure
                Path(
                    f"../../synth_env/examples/{env_name}/agent_demos/src/evals/{env_name}/{run_id}/traces/{trace_id}.json"
                ),
                Path(
                    f"../synth_env/examples/{env_name}/agent_demos/src/evals/{env_name}/{run_id}/traces/{trace_id}.json"
                ),
                Path(
                    f"../../../synth_env/examples/{env_name}/agent_demos/src/evals/{env_name}/{run_id}/traces/{trace_id}.json"
                ),
                # Sokoban specific paths
                Path(f"../../evals/{env_name}_proper/{run_id}/traces/{trace_id}.json"),
                Path(f"../evals/{env_name}_proper/{run_id}/traces/{trace_id}.json"),
                Path(
                    f"../../../evals/{env_name}_proper/{run_id}/traces/{trace_id}.json"
                ),
                # NetHack/MiniGrid paths
                Path(f"../../evals/{env_name}/{run_id}/traces/{trace_id}.json"),
                Path(f"../evals/{env_name}/{run_id}/traces/{trace_id}.json"),
                Path(f"../../../evals/{env_name}/{run_id}/traces/{trace_id}.json"),
            ]

            for path in possible_paths:
                if path.exists():
                    with open(path, "r") as f:
                        return json.load(f)

        # If we still can't find it, silently return None (don't show annoying warnings)
        return None

    except Exception as e:
        st.error(f"Error loading trace data: {e}")
        return None


def discover_evaluation_runs():
    """Discover all evaluation runs in the evals directory."""
    # Look for evaluation runs in the standard locations
    search_paths = [
        # Crafter evaluation runs
        Path(
            "../synth_env/examples/crafter_classic/agent_demos/src/evals/crafter/run_*"
        ),
        Path(
            "../../synth_env/examples/crafter_classic/agent_demos/src/evals/crafter/run_*"
        ),
        Path(
            "../../../synth_env/examples/crafter_classic/agent_demos/src/evals/crafter/run_*"
        ),
        Path("../src/evals/crafter/run_*"),
        Path("../../src/evals/crafter/run_*"),
        Path("../../../src/evals/crafter/run_*"),
        # Sokoban evaluation runs
        Path("../synth_env/examples/sokoban/agent_demos/src/evals/sokoban/run_*"),
        Path("../../synth_env/examples/sokoban/agent_demos/src/evals/sokoban/run_*"),
        Path("../../../synth_env/examples/sokoban/agent_demos/src/evals/sokoban/run_*"),
        # Sokoban proper evaluation runs (with agent reasoning)
        Path("../evals/sokoban_proper/run_*"),
        Path("../../evals/sokoban_proper/run_*"),
        Path("../../../evals/sokoban_proper/run_*"),
        # NetHack evaluation runs
        Path("../evals/nethack/run_*"),
        Path("../../evals/nethack/run_*"),
        Path("../../../evals/nethack/run_*"),
        # MiniGrid evaluation runs
        Path("../evals/minigrid/run_*"),
        Path("../../evals/minigrid/run_*"),
        Path("../../../evals/minigrid/run_*"),
    ]

    eval_runs = []
    for search_path in search_paths:
        for run_dir in glob.glob(str(search_path)):
            run_path = Path(run_dir)
            if run_path.is_dir():
                summary_file = run_path / "evaluation_summary.json"
                traces_dir = run_path / "traces"

                # Accept runs with either summary files OR just traces directory
                if (summary_file.exists() and traces_dir.exists()) or (
                    traces_dir.exists() and len(list(traces_dir.glob("*.json"))) > 0
                ):
                    eval_runs.append(run_path)

    return eval_runs


def import_evaluation_run_streamlit(run_path: Path, conn):
    """Import evaluation run using DuckDB connection (Streamlit version)."""
    try:
        # Load evaluation summary if it exists
        summary_file = run_path / "evaluation_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
            models_evaluated = summary.get("models_evaluated", ["unknown"])
            difficulties_evaluated = summary.get("difficulties_evaluated", ["unknown"])
            num_trajectories = summary.get("evaluation_metadata", {}).get(
                "num_trajectories", 2
            )
        else:
            # No summary file, create defaults
            summary = {}
            models_evaluated = ["unknown"]
            difficulties_evaluated = ["unknown"]
            # Count actual trace files
            traces_dir = run_path / "traces"
            num_trajectories = (
                len(list(traces_dir.glob("*.json"))) if traces_dir.exists() else 0
            )

        # Check if already imported
        existing = conn.execute(
            "SELECT COUNT(*) FROM evaluations WHERE run_id = ?", [run_path.name]
        ).fetchone()[0]
        if existing > 0:
            return f"⚠️ {run_path.name} already imported"

        # Get or create environment
        env_name = "crafter"  # default fallback
        path_lower = str(run_path).lower()
        if "sokoban" in path_lower:
            env_name = "sokoban"
        elif "nethack" in path_lower:
            env_name = "nethack"
        elif "minigrid" in path_lower:
            env_name = "minigrid"
        elif "crafter" in path_lower:
            env_name = "crafter"

        # Get environment ID
        env_result = conn.execute(
            "SELECT id FROM environments WHERE name = ?", [env_name]
        ).fetchone()
        if not env_result:
            return f"❌ Environment {env_name} not found"
        env_id = env_result[0]

        # Generate evaluation ID
        eval_id = hash(run_path.name) % (2**31)

        # Insert evaluation
        conn.execute(
            """
            INSERT INTO evaluations (
                id, env_id, run_id, timestamp, models_evaluated, difficulties_evaluated,
                num_trajectories, success_rate, avg_achievements, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                eval_id,
                env_id,
                run_path.name,
                datetime.now(),
                json.dumps(models_evaluated),
                json.dumps(difficulties_evaluated),
                num_trajectories,
                0.0,
                0.0,
                json.dumps(summary),
            ],
        )

        # Import trajectories
        traces_dir = run_path / "traces"
        trace_files = list(traces_dir.glob("*.json"))
        imported_count = 0

        for trace_file in trace_files:
            trajectory_id = trace_file.stem
            traj_id = hash(trajectory_id) % (2**31)

            # Load trace data to extract metadata
            with open(trace_file, "r") as f:
                trace_data = json.load(f)

            # Extract metadata (simplified version)
            trace_metadata = trace_data.get("trace", {}).get("metadata", {})
            reward_signals = trace_data.get("dataset", {}).get("reward_signals", [])

            # Get final stats
            partitions = trace_data.get("trace", {}).get("partition", [])
            final_reward = 0.0
            num_steps = 0
            terminated = False
            total_achievements = 0
            success = False
            boxes_solved = 0
            total_boxes = 0

            # Extract metadata from trace metadata (common for all environments)
            if trace_metadata:
                final_reward = trace_metadata.get("final_reward", 0.0)
                num_steps = trace_metadata.get("num_steps", 0)
                success = trace_metadata.get("success", False)

                # Sokoban-specific metadata
                if env_name == "sokoban":
                    boxes_solved = trace_metadata.get("boxes_solved", 0)
                    total_boxes = trace_metadata.get("total_boxes", 0)
                    terminated = success  # For Sokoban, success means terminated
                else:
                    # For other environments, try to extract from partitions
                    if partitions:
                        last_partition = partitions[-1]
                        events = last_partition.get("events", [])
                        if events:
                            last_event = events[-1]
                            env_steps = last_event.get("environment_compute_steps", [])
                            if env_steps:
                                last_env_step = env_steps[-1]
                                outputs = last_env_step.get("compute_output", [{}])[
                                    0
                                ].get("outputs", {})
                                final_reward = outputs.get("total_reward", final_reward)
                                num_steps = outputs.get("num_steps", num_steps)
                                terminated = outputs.get("terminated", False)

                            event_metadata = last_event.get("event_metadata", {})
                            total_achievements = event_metadata.get(
                                "total_achievements", 0
                            )

            if reward_signals:
                final_reward = reward_signals[0].get("reward", final_reward)

            # Insert trajectory
            conn.execute(
                """
                INSERT INTO trajectories (
                    id, eval_id, trace_id, model_name, difficulty, seed, success,
                    final_reward, num_steps, achievements, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    traj_id,
                    eval_id,
                    trajectory_id,
                    trace_metadata.get(
                        "model_name",
                        models_evaluated[0] if models_evaluated else "unknown",
                    ),
                    trace_metadata.get(
                        "difficulty",
                        difficulties_evaluated[0]
                        if difficulties_evaluated
                        else "unknown",
                    ),
                    trace_metadata.get("seed"),
                    success,
                    final_reward,
                    num_steps,
                    json.dumps(list(range(total_achievements))),
                    json.dumps(
                        {
                            "trace_metadata": trace_metadata,
                            "total_achievements": total_achievements,
                            "boxes_solved": boxes_solved,
                            "total_boxes": total_boxes,
                            "environment_name": env_name,
                        }
                    ),
                ],
            )

            # Insert trace record
            trace_id = hash(f"{trajectory_id}_trace") % (2**31)
            conn.execute(
                """
                INSERT INTO traces (
                    id, trajectory_id, parquet_path, trace_format, size_bytes
                ) VALUES (?, ?, ?, ?, ?)
            """,
                [trace_id, traj_id, str(trace_file), "json", trace_file.stat().st_size],
            )

            imported_count += 1

        return f"✅ Imported {run_path.name}: {imported_count} trajectories"

    except Exception as e:
        return f"❌ Error importing {run_path.name}: {str(e)}"


def sync_traces(conn):
    """Discover and import new evaluation runs, and clean up stale entries."""
    # First, clean up stale entries
    cleanup_results = cleanup_stale_traces(conn)

    discovered_runs = discover_evaluation_runs()

    if not discovered_runs:
        return ["No evaluation runs found in expected directories"] + cleanup_results

    results = []
    for run_path in discovered_runs:
        result = import_evaluation_run_streamlit(run_path, conn)
        results.append(result)

    return results + cleanup_results


def cleanup_stale_traces(conn):
    """Remove database entries for traces that no longer exist on filesystem."""
    results = []

    # Get all traces with their file paths
    query = """
    SELECT t.id, t.parquet_path, traj.trace_id, traj.eval_id, e.run_id, env.name as env_name
    FROM traces t
    JOIN trajectories traj ON t.trajectory_id = traj.id
    JOIN evaluations e ON traj.eval_id = e.id
    JOIN environments env ON e.env_id = env.id
    """

    traces = conn.execute(query).fetchall()
    stale_trace_ids = []
    stale_trajectory_ids = []
    stale_eval_ids = set()

    for trace_id, file_path, trace_uuid, eval_id, run_id, env_name in traces:
        trace_file = Path(file_path)

        # Check if trace file exists
        if not trace_file.exists():
            # Try fallback locations
            possible_paths = [
                Path(f"../../src/evals/{env_name}/{run_id}/traces/{trace_uuid}.json"),
                Path(f"../src/evals/{env_name}/{run_id}/traces/{trace_uuid}.json"),
                Path(
                    f"../../../src/evals/{env_name}/{run_id}/traces/{trace_uuid}.json"
                ),
                Path(
                    f"../../evals/{env_name}_proper/{run_id}/traces/{trace_uuid}.json"
                ),
                Path(f"../evals/{env_name}_proper/{run_id}/traces/{trace_uuid}.json"),
                Path(
                    f"../../../evals/{env_name}_proper/{run_id}/traces/{trace_uuid}.json"
                ),
                Path(f"../../evals/{env_name}/{run_id}/traces/{trace_uuid}.json"),
                Path(f"../evals/{env_name}/{run_id}/traces/{trace_uuid}.json"),
                Path(f"../../../evals/{env_name}/{run_id}/traces/{trace_uuid}.json"),
            ]

            found = False
            for path in possible_paths:
                if path.exists():
                    found = True
                    break

            if not found:
                stale_trace_ids.append(trace_id)
                stale_trajectory_ids.append(trace_uuid)
                stale_eval_ids.add(eval_id)

    # Remove stale traces
    if stale_trace_ids:
        conn.execute(
            f"DELETE FROM traces WHERE id IN ({','.join(['?'] * len(stale_trace_ids))})",
            stale_trace_ids,
        )
        results.append(f"🧹 Removed {len(stale_trace_ids)} stale trace entries")

    # Remove stale trajectories
    if stale_trajectory_ids:
        conn.execute(
            f"DELETE FROM trajectories WHERE trace_id IN ({','.join(['?'] * len(stale_trajectory_ids))})",
            stale_trajectory_ids,
        )
        results.append(
            f"🧹 Removed {len(stale_trajectory_ids)} stale trajectory entries"
        )

    # Remove evaluations with no remaining trajectories
    for eval_id in stale_eval_ids:
        remaining_count = conn.execute(
            "SELECT COUNT(*) FROM trajectories WHERE eval_id = ?", [eval_id]
        ).fetchone()[0]
        if remaining_count == 0:
            conn.execute("DELETE FROM evaluations WHERE id = ?", [eval_id])
            results.append(f"🧹 Removed empty evaluation {eval_id}")

    if not results:
        results.append("✅ No stale entries found")

    return results


# Environment-specific processors
def process_crafter_trace(trace_data: Dict) -> Dict[str, Any]:
    """Process Crafter trace data."""
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})

    processed = {
        "model_name": metadata.get("model_name", "Unknown"),
        "difficulty": metadata.get("difficulty", "Unknown"),
        "total_reward": dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
        if dataset.get("reward_signals")
        else 0.0,
        "turns": [],
    }

    partitions = trace.get("partition", [])
    for i, partition in enumerate(partitions[:10]):  # Limit for performance
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        env_steps = event.get("environment_compute_steps", [])

        turn_data = {
            "turn_number": i + 1,
            "images": [],
            "actions": [],
            "stats": None,
            "achievements": event.get("event_metadata", {}).get("new_achievements", []),
        }

        for step_idx, step in enumerate(env_steps):
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})

            # Extract image
            if "image_base64" in outputs:
                img_data = outputs["image_base64"]
                if not img_data.startswith("data:"):
                    img_data = f"data:image/png;base64,{img_data}"

                action_idx = outputs.get("action_index", -1)
                action_name = get_crafter_action_name(action_idx)

                turn_data["images"].append(
                    {
                        "data_url": img_data,
                        "caption": f"Step {step_idx}: {action_name}",
                        "step_number": step_idx,
                    }
                )

            # Extract action
            action_idx = outputs.get("action_index", -1)
            if action_idx >= 0:
                action_name = get_crafter_action_name(action_idx)
                turn_data["actions"].append({"name": action_name, "index": action_idx})

            # Extract stats
            if "player_stats" in outputs:
                player_stats = outputs["player_stats"]
                turn_data["stats"] = {
                    "health": player_stats.get("health", 0),
                    "food": player_stats.get("food", 0),
                    "drink": player_stats.get("drink", 0),
                }

        processed["turns"].append(turn_data)

    return processed


def process_nethack_trace(trace_data: Dict) -> Dict[str, Any]:
    """Process NetHack trace data."""
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})

    processed = {
        "model_name": metadata.get("model_name", "Unknown"),
        "difficulty": metadata.get("difficulty", "Unknown"),
        "total_reward": dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
        if dataset.get("reward_signals")
        else 0.0,
        "turns": [],
    }

    # Partitions can be at top level or under trace
    partitions = trace_data.get("partition", trace.get("partition", []))
    for i, partition in enumerate(partitions[:10]):
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        env_steps = event.get("environment_compute_steps", [])

        turn_data = {"turn_number": i + 1, "messages": [], "actions": [], "stats": None}

        for step in env_steps:
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})

            # Extract messages
            if "message" in outputs:
                turn_data["messages"].append(outputs["message"])

            # Extract stats
            if "blstats" in outputs and len(outputs["blstats"]) >= 25:
                blstats = outputs["blstats"]
                turn_data["stats"] = {
                    "hp": blstats[10],
                    "max_hp": blstats[11],
                    "level": blstats[18],
                    "gold": blstats[13],
                    "ac": blstats[17],
                }

            # Extract action
            if "action" in outputs:
                action = outputs["action"]
                if isinstance(action, int):
                    turn_data["actions"].append(
                        {"name": f"action_{action}", "index": action}
                    )

        processed["turns"].append(turn_data)

    return processed


def process_sokoban_trace(trace_data: Dict) -> Dict[str, Any]:
    """Process Sokoban trace data."""
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})

    processed = {
        "model_name": metadata.get("model_name", "Unknown"),
        "difficulty": metadata.get("difficulty", "Unknown"),
        "total_reward": dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
        if dataset.get("reward_signals")
        else 0.0,
        "success": metadata.get("success", False),
        "num_steps": metadata.get("num_steps", 0),
        "boxes_solved": metadata.get("boxes_solved", 0),
        "total_boxes": metadata.get("total_boxes", 0),
        "turns": [],
    }

    partitions = trace.get("partition", [])

    for i, partition in enumerate(partitions):  # Process all partitions
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        turn_data_raw = event.get("event_metadata", {}).get("turn_data", {})

        turn_data = {
            "turn_number": turn_data_raw.get("turn_number", i + 1),
            "room_text": turn_data_raw.get("room_text", ""),
            "player_position": turn_data_raw.get("player_position", [0, 0]),
            "boxes_on_target": turn_data_raw.get("boxes_on_target", 0),
            "num_steps": turn_data_raw.get("num_steps", 0),
            "action_taken": turn_data_raw.get("action_taken", -1),
            "action_name": turn_data_raw.get("action_name", "initial"),
            "reward": turn_data_raw.get("reward", 0.0),
            "total_reward": turn_data_raw.get("total_reward", 0.0),
            "terminated": turn_data_raw.get("terminated", False),
            "truncated": turn_data_raw.get("truncated", False),
        }

        processed["turns"].append(turn_data)

    return processed


def process_minigrid_trace(trace_data: Dict) -> Dict[str, Any]:
    """Process MiniGrid trace data."""
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})

    processed = {
        "model_name": metadata.get("model_name", "Unknown"),
        "env_name": metadata.get("env_name", "Unknown MiniGrid Environment"),
        "difficulty": metadata.get("difficulty", "Unknown"),
        "total_reward": dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
        if dataset.get("reward_signals")
        else 0.0,
        "success": metadata.get("success", False),
        "turns": [],
    }

    # Partitions can be at top level or under trace
    partitions = trace_data.get("partition", trace.get("partition", []))
    for i, partition in enumerate(partitions[:15]):  # MiniGrid episodes can be short
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        env_steps = event.get("environment_compute_steps", [])

        turn_data = {
            "turn_number": i + 1,
            "images": [],
            "actions": [],
            "rewards": [],
            "mission": "",
        }

        for step in env_steps:
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})

            # Extract observation
            if "observation" in outputs:
                obs = outputs["observation"]
                if isinstance(obs, dict):
                    if "mission" in obs:
                        turn_data["mission"] = obs["mission"]
                    if "image" in obs:
                        img_data = obs["image"]
                        if isinstance(img_data, str):
                            if not img_data.startswith("data:"):
                                img_data = f"data:image/png;base64,{img_data}"
                            turn_data["images"].append(img_data)
                    elif "image_base64" in obs:
                        turn_data["images"].append(
                            f"data:image/png;base64,{obs['image_base64']}"
                        )

            # Extract image directly
            if "image_base64" in outputs:
                turn_data["images"].append(
                    f"data:image/png;base64,{outputs['image_base64']}"
                )
            elif "image" in outputs and isinstance(outputs["image"], str):
                img_data = outputs["image"]
                if not img_data.startswith("data:"):
                    img_data = f"data:image/png;base64,{img_data}"
                turn_data["images"].append(img_data)

            # Extract action
            if "action" in outputs:
                action = outputs["action"]
                if isinstance(action, int):
                    action_name = get_minigrid_action_name(action)
                    turn_data["actions"].append({"name": action_name, "index": action})

            # Extract reward
            if "reward" in outputs:
                turn_data["rewards"].append(float(outputs["reward"]))

        processed["turns"].append(turn_data)

    return processed


def get_crafter_action_name(action_idx: int) -> str:
    """Map Crafter action index to name."""
    action_names = {
        -1: "initial state",
        0: "noop",
        1: "move_left",
        2: "move_right",
        3: "move_up",
        4: "move_down",
        5: "do",
        6: "sleep",
        7: "place_stone",
        8: "place_table",
        9: "place_furnace",
        10: "place_plant",
        11: "make_wood_pickaxe",
        12: "make_stone_pickaxe",
        13: "make_iron_pickaxe",
        14: "make_wood_sword",
        15: "make_stone_sword",
        16: "make_iron_sword",
    }
    return action_names.get(action_idx, f"unknown_{action_idx}")


def get_minigrid_action_name(action_idx: int) -> str:
    """Map MiniGrid action index to name."""
    action_names = {
        0: "left",
        1: "right",
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done",
    }
    return action_names.get(action_idx, f"action_{action_idx}")


# Visualization functions
def render_crafter_trace(processed_trace: Dict):
    """Render Crafter trace visualization."""
    # Collect all achievements across turns
    all_achievements = []
    for turn in processed_trace["turns"]:
        all_achievements.extend(turn.get("achievements", []))

    # Ultra-compact info line
    st.markdown(
        f"🎮 **Turns:** {len(processed_trace['turns'])} | **Reward:** {processed_trace['total_reward']:.3f} | **Difficulty:** {processed_trace['difficulty']}"
    )

    # Achievements section
    if all_achievements:
        st.markdown(
            f"🏆 **Achievements ({len(all_achievements)}):** "
            + " | ".join(
                [f"**{ach.replace('_', ' ').title()}**" for ach in all_achievements]
            )
        )
    else:
        st.markdown("🏆 **Achievements (0):** None")

    # Compact turn selector
    if len(processed_trace["turns"]) > 1:
        selected_turn_idx = (
            st.slider(
                "Turn",
                min_value=1,
                max_value=len(processed_trace["turns"]),
                value=1,
                step=1,
                format="Turn %d",
            )
            - 1
        )  # Convert to 0-based index
    else:
        selected_turn_idx = 0

    # Display selected turn
    if 0 <= selected_turn_idx < len(processed_trace["turns"]):
        render_crafter_turn(processed_trace["turns"][selected_turn_idx])

    # Get the original trace data from the database
    conn = get_db_connection()
    trajectory_id = st.session_state.get("current_trajectory_id")
    if trajectory_id:
        raw_trace_data = load_trace_data(conn, trajectory_id)
        if raw_trace_data:
            # Show event details for the selected turn
            show_synth_sdk_event_details(raw_trace_data, selected_turn_idx)
        else:
            st.error("Could not load raw trace data")
    else:
        st.warning("No trajectory ID available")

    # Compact debug view - collapsible (show only current event)
    with st.expander("🔍 Raw JSON Data", expanded=False):
        if trajectory_id and raw_trace_data:
            # Show only the current event data
            if "trace" in raw_trace_data and "partition" in raw_trace_data["trace"]:
                partitions = raw_trace_data["trace"]["partition"]
                if selected_turn_idx < len(partitions):
                    st.json(partitions[selected_turn_idx])
                else:
                    st.warning(f"Turn {selected_turn_idx} not found")
            elif "partition" in raw_trace_data:
                partitions = raw_trace_data["partition"]
                if selected_turn_idx < len(partitions):
                    st.json(partitions[selected_turn_idx])
                else:
                    st.warning(f"Turn {selected_turn_idx} not found")
            else:
                st.json(raw_trace_data)
        else:
            st.warning("No raw trace data available")


def render_nethack_trace(processed_trace: Dict):
    """Render NetHack trace visualization."""
    st.header(f"🏰 NetHack Trace - {processed_trace['model_name']}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Score", f"{processed_trace['total_reward']:.0f}")
    with col2:
        st.metric("Difficulty", processed_trace["difficulty"])
    with col3:
        st.metric("Total Turns", len(processed_trace["turns"]))

    st.divider()

    # Turn selector
    if len(processed_trace["turns"]) > 1:
        selected_turn_idx = (
            st.slider(
                "Turn",
                min_value=1,
                max_value=len(processed_trace["turns"]),
                value=1,
                step=1,
                format="Turn %d",
            )
            - 1
        )  # Convert to 0-based index
    else:
        selected_turn_idx = 0

    # Display selected turn
    if 0 <= selected_turn_idx < len(processed_trace["turns"]):
        render_nethack_turn(processed_trace["turns"][selected_turn_idx])

    # Show all turns in compact view
    if st.checkbox("Show All Turns (Compact View)"):
        st.subheader("All Turns Overview")
        for i, turn in enumerate(
            processed_trace["turns"][:10]
        ):  # Limit to first 10 turns
            with st.expander(f"Turn {turn['turn_number']}", expanded=False):
                render_nethack_turn_compact(turn)


def render_sokoban_trace(processed_trace: Dict):
    """Render Sokoban trace visualization."""
    # Ultra-compact info line
    st.markdown(
        f"🎮 **Turns:** {len(processed_trace['turns'])} | **Boxes Solved:** {processed_trace['boxes_solved']}/{processed_trace['total_boxes']}"
    )

    # Compact turn selector
    if len(processed_trace["turns"]) > 1:
        selected_turn_idx = (
            st.slider(
                "Turn",
                min_value=1,
                max_value=len(processed_trace["turns"]),
                value=1,
                step=1,
                format="Turn %d",
            )
            - 1
        )  # Convert to 0-based index
    else:
        selected_turn_idx = 0

    # Display selected turn
    if 0 <= selected_turn_idx < len(processed_trace["turns"]):
        render_sokoban_turn(processed_trace["turns"][selected_turn_idx])

    # Get the original trace data from the database
    conn = get_db_connection()
    trajectory_id = st.session_state.get("current_trajectory_id")
    if trajectory_id:
        raw_trace_data = load_trace_data(conn, trajectory_id)
        if raw_trace_data:
            # Show event details for the selected turn
            show_synth_sdk_event_details(raw_trace_data, selected_turn_idx)
        else:
            st.error("Could not load raw trace data")
    else:
        st.warning("No trajectory ID available")

    # Compact debug view - collapsible (show only current event)
    with st.expander("🔍 Raw JSON Data", expanded=False):
        if trajectory_id and raw_trace_data:
            # Show only the current event data
            if "trace" in raw_trace_data and "partition" in raw_trace_data["trace"]:
                partitions = raw_trace_data["trace"]["partition"]
                if selected_turn_idx < len(partitions):
                    st.json(partitions[selected_turn_idx])
                else:
                    st.warning(f"Turn {selected_turn_idx} not found")
            elif "partition" in raw_trace_data:
                partitions = raw_trace_data["partition"]
                if selected_turn_idx < len(partitions):
                    st.json(partitions[selected_turn_idx])
                else:
                    st.warning(f"Turn {selected_turn_idx} not found")
            else:
                st.json(raw_trace_data)
        else:
            st.warning("No raw trace data available")


def render_minigrid_trace(processed_trace: Dict):
    """Render MiniGrid trace visualization."""
    st.header(f"🟦 MiniGrid - {processed_trace['env_name']}")
    st.subheader(f"Model: {processed_trace['model_name']}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reward", f"{processed_trace['total_reward']:.2f}")
    with col2:
        st.metric("Success", "✅" if processed_trace["success"] else "❌")
    with col3:
        st.metric("Total Steps", len(processed_trace["turns"]))

    st.divider()

    # Turn selector
    if len(processed_trace["turns"]) > 1:
        selected_turn_idx = (
            st.slider(
                "Step",
                min_value=1,
                max_value=len(processed_trace["turns"]),
                value=1,
                step=1,
                format="Step %d",
            )
            - 1
        )  # Convert to 0-based index
    else:
        selected_turn_idx = 0

    # Display selected turn
    if 0 <= selected_turn_idx < len(processed_trace["turns"]):
        render_minigrid_turn(processed_trace["turns"][selected_turn_idx])

    # Show all turns in compact view
    if st.checkbox("Show All Steps (Compact View)"):
        st.subheader("All Steps Overview")
        for i, turn in enumerate(
            processed_trace["turns"][:15]
        ):  # Show more steps for MiniGrid
            with st.expander(f"Step {turn['turn_number']}", expanded=False):
                render_minigrid_turn_compact(turn)


def render_generic_trace(trace_data: Dict):
    """Render generic trace (raw JSON)."""
    st.header("📄 Generic Trace")
    st.json(trace_data)


def show_synth_sdk_event_details(raw_trace_data: Dict, turn_index: int):
    """Show detailed Synth SDK event information for a specific turn."""
    try:
        # Navigate to the correct event based on turn index
        if "trace" in raw_trace_data and "partition" in raw_trace_data["trace"]:
            partitions = raw_trace_data["trace"]["partition"]

            if turn_index < len(partitions):
                partition = partitions[turn_index]
                events = partition.get("events", [])

                if events:
                    event = events[0]  # Usually one event per partition

                    st.write(f"**Event Type:** {event.get('event_type', 'Unknown')}")

                    # Show agent compute step if available
                    if "agent_compute_step" in event and event["agent_compute_step"]:
                        agent_step = event["agent_compute_step"]

                        # Show message counts (model already shown above)
                        # Count messages
                        input_msgs = 0
                        if (
                            "compute_input" in agent_step
                            and agent_step["compute_input"]
                        ):
                            for input_item in agent_step["compute_input"]:
                                if "messages" in input_item:
                                    input_msgs += len(input_item["messages"])

                        # Count output messages
                        output_msgs = 0
                        if (
                            "compute_output" in agent_step
                            and agent_step["compute_output"]
                        ):
                            for output_item in agent_step["compute_output"]:
                                if "messages" in output_item:
                                    output_msgs += len(output_item["messages"])

                        agent_summary = (
                            f"🔍 **Agent Summary:** {input_msgs} in | {output_msgs} out"
                        )
                        st.markdown(agent_summary)

                        # Show compute input (messages sent to agent) - collapsible
                        if (
                            "compute_input" in agent_step
                            and agent_step["compute_input"]
                        ):
                            with st.expander("📨 Input Messages", expanded=False):
                                for i, input_item in enumerate(
                                    agent_step["compute_input"]
                                ):
                                    if "messages" in input_item:
                                        for msg in input_item["messages"]:
                                            role = msg.get("role", "unknown")
                                            content = msg.get("content", "")

                                            # Use different styling for different roles
                                            if role == "system":
                                                st.info(f"**System:** {content}")
                                            elif role == "user":
                                                st.success(f"**User:** {content}")
                                            else:
                                                st.write(
                                                    f"**{role.title()}:** {content}"
                                                )

                        # Show compute output (agent's response) - collapsible
                        if (
                            "compute_output" in agent_step
                            and agent_step["compute_output"]
                        ):
                            with st.expander("🤖 Agent Response", expanded=False):
                                for i, output_item in enumerate(
                                    agent_step["compute_output"]
                                ):
                                    if "messages" in output_item:
                                        for msg in output_item["messages"]:
                                            role = msg.get("role", "unknown")
                                            content = msg.get("content", "")

                                            if role == "assistant":
                                                st.warning(f"**Agent:** {content}")

                                                # Show tool calls if present
                                                if (
                                                    "tool_calls" in msg
                                                    and msg["tool_calls"]
                                                ):
                                                    st.write("**🔧 Tool Calls:**")
                                                    for tool_call in msg["tool_calls"]:
                                                        func_name = tool_call.get(
                                                            "function", {}
                                                        ).get("name", "unknown")
                                                        args = tool_call.get(
                                                            "function", {}
                                                        ).get("arguments", "")
                                                        st.code(
                                                            f"{func_name}({args})",
                                                            language="json",
                                                        )

                                            elif role == "tool":
                                                tool_call_id = msg.get(
                                                    "tool_call_id", "unknown"
                                                )
                                                st.error(
                                                    f"**🛠️ Tool ({tool_call_id}):** {content}"
                                                )

                                            else:
                                                st.write(
                                                    f"**{role.title()}:** {content}"
                                                )
                    else:
                        st.warning(
                            "⚠️ No agent compute step found - this trace is missing agent reasoning!"
                        )

                    # Show environment compute steps - collapsible
                    if (
                        "environment_compute_steps" in event
                        and event["environment_compute_steps"]
                    ):
                        with st.expander("🌍 Environment Response", expanded=False):
                            for i, env_step in enumerate(
                                event["environment_compute_steps"]
                            ):
                                if (
                                    "compute_output" in env_step
                                    and env_step["compute_output"]
                                ):
                                    for output in env_step["compute_output"]:
                                        if "outputs" in output:
                                            st.json(output["outputs"])

                    # Show event metadata - collapsible
                    if "event_metadata" in event and event["event_metadata"]:
                        with st.expander("📋 Event Metadata", expanded=False):
                            st.json(event["event_metadata"])

                else:
                    st.warning(f"No events found in partition {turn_index}")
            else:
                st.warning(
                    f"Turn {turn_index} not found (only {len(partitions)} partitions available)"
                )

        # Handle direct partition format (like NetHack traces)
        elif "partition" in raw_trace_data:
            partitions = raw_trace_data["partition"]

            if turn_index < len(partitions):
                partition = partitions[turn_index]
                events = partition.get("events", [])

                if events:
                    event = events[0]
                    st.write(f"**Event Type:** {event.get('event_type', 'Unknown')}")

                    # Show the full event structure
                    st.subheader("📋 Full Event Data")
                    st.json(event)
                else:
                    st.warning(f"No events found in partition {turn_index}")
            else:
                st.warning(
                    f"Turn {turn_index} not found (only {len(partitions)} partitions available)"
                )

        else:
            st.warning("Trace data does not have the expected Synth SDK structure")

    except Exception as e:
        st.error(f"Error showing event details: {str(e)}")
        st.write("Raw trace structure:")
        st.json(raw_trace_data)


def render_trajectory_details(trajectory: Dict, conn=None):
    """Render trajectory details."""
    # Ultra-compact trajectory info - single line
    trajectory_info = f"🎯 **{trajectory['model_name']}** | {trajectory['difficulty']} | Seed: {trajectory['seed']} | ID: {trajectory['trace_id'][:8]}... | Reward: {trajectory['final_reward']:.3f} | {'✅' if trajectory['success'] else '❌'} | Steps: {trajectory['num_steps']}"
    st.markdown(trajectory_info)

    # Load and display trace visualization immediately
    if conn:
        render_trace_visualization(trajectory, conn)
    else:
        st.info(
            "💡 Full trace visualization will be available when trace data processing is implemented."
        )


def render_trace_visualization(trajectory: Dict, conn):
    """Render step-by-step trace visualization."""
    # Store trajectory ID in session state for raw trace access
    st.session_state["current_trajectory_id"] = trajectory["id"]

    # Load trace data
    trace_data = load_trace_data(conn, trajectory["id"])

    if not trace_data:
        st.error("Could not load trace data")
        return

    # Determine environment type from trace data or trajectory
    env_type = "crafter"  # default
    meta_env_name = ""
    if "environment_name" in trajectory:
        meta_env_name = str(trajectory.get("environment_name", ""))
    elif trajectory.get("metadata"):
        meta_env_name = str(trajectory["metadata"].get("environment_name", ""))

    if "sokoban" in meta_env_name.lower():
        env_type = "sokoban"
    elif "nethack" in meta_env_name.lower():
        env_type = "nethack"
    elif "minigrid" in meta_env_name.lower():
        env_type = "minigrid"

    # Process trace data based on environment type
    if env_type == "sokoban":
        processed_trace = process_sokoban_trace(trace_data)
    elif env_type == "nethack":
        processed_trace = process_nethack_trace(trace_data)
    elif env_type == "minigrid":
        processed_trace = process_minigrid_trace(trace_data)
    else:
        processed_trace = process_crafter_trace(trace_data)

    if not processed_trace["turns"]:
        st.info("No trace turns found")
        return

    # Render based on environment type
    if env_type == "sokoban":
        render_sokoban_trace(processed_trace)
    elif env_type == "nethack":
        render_nethack_trace(processed_trace)
    elif env_type == "minigrid":
        render_minigrid_trace(processed_trace)
    else:
        render_crafter_trace(processed_trace)


def render_crafter_turn(turn_data: Dict):
    """Render a detailed view of a single Crafter turn."""
    st.subheader(f"Turn {turn_data['turn_number']}")

    # Game state visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        # Images
        if turn_data["images"]:
            st.subheader("🎮 Game States")
            cols = st.columns(min(len(turn_data["images"]), 3))
            for i, img in enumerate(turn_data["images"][:3]):  # Show max 3 images
                with cols[i % 3]:
                    try:
                        st.image(img["data_url"], caption=img["caption"], width=200)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")

        # Actions taken
        if turn_data["actions"]:
            st.subheader("Actions")
            action_text = " → ".join(
                [action["name"] for action in turn_data["actions"]]
            )
            st.text(action_text)

    with col2:
        # Player stats
        if turn_data["stats"]:
            st.subheader("📊 Player Stats")
            st.metric("❤️ Health", turn_data["stats"]["health"])
            st.metric("🍖 Food", turn_data["stats"]["food"])
            st.metric("💧 Drink", turn_data["stats"]["drink"])

        # Achievements
        if turn_data["achievements"]:
            st.subheader("🏆 New Achievements")
            for achievement in turn_data["achievements"]:
                # Display achievement name (achievements are strings like "collect_wood")
                if isinstance(achievement, str):
                    st.success(f"🏆 {achievement.replace('_', ' ').title()}")
                else:
                    # Handle numeric achievements (legacy format)
                    st.success(f"🏆 Achievement {achievement}")


def render_crafter_turn_compact(turn_data: Dict):
    """Render a compact view of a Crafter turn."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if turn_data["stats"]:
            st.write(f"**HP:** {turn_data['stats']['health']}")
            st.write(f"**Food:** {turn_data['stats']['food']}")

    with col2:
        if turn_data["actions"]:
            actions = [action["name"] for action in turn_data["actions"]]
            st.write(f"**Actions:** {', '.join(actions[:3])}")
        if turn_data["achievements"]:
            st.write(f"**🏆 Achievements:** {len(turn_data['achievements'])}")

    with col3:
        if turn_data["images"]:
            # Show first image only
            st.image(turn_data["images"][0]["data_url"], width=150)


def render_nethack_turn(turn_data: Dict):
    """Render a detailed view of a single NetHack turn."""
    st.subheader(f"Turn {turn_data['turn_number']}")

    # Character stats
    if turn_data["stats"]:
        st.subheader("Character Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "HP", f"{turn_data['stats']['hp']}/{turn_data['stats']['max_hp']}"
            )
        with col2:
            st.metric("Level", turn_data["stats"]["level"])
        with col3:
            st.metric("Gold", turn_data["stats"]["gold"])
        with col4:
            st.metric("AC", turn_data["stats"]["ac"])

    # Actions taken
    if turn_data["actions"]:
        st.subheader("Actions")
        action_text = " → ".join(
            [f"{action['name']} ({action['index']})" for action in turn_data["actions"]]
        )
        st.text(action_text)

    # Game messages
    if turn_data["messages"]:
        st.subheader("Game Messages")
        for message in turn_data["messages"][-5:]:  # Show last 5 messages
            st.text(message)


def render_nethack_turn_compact(turn_data: Dict):
    """Render a compact view of a NetHack turn."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if turn_data["stats"]:
            st.write(
                f"**HP:** {turn_data['stats']['hp']}/{turn_data['stats']['max_hp']}"
            )
            st.write(f"**Level:** {turn_data['stats']['level']}")

    with col2:
        if turn_data["actions"]:
            actions = [action["name"] for action in turn_data["actions"]]
            st.write(f"**Actions:** {', '.join(actions[:2])}")
        st.write(
            f"**Gold:** {turn_data['stats']['gold']}" if turn_data["stats"] else ""
        )

    with col3:
        if turn_data["messages"]:
            # Show most recent message
            st.write(f"**Message:** {turn_data['messages'][-1][:50]}...")


def render_sokoban_turn(turn_data: Dict):
    """Render a detailed view of a single Sokoban turn."""
    # Ultra-compact turn header
    action_text = (
        f" | **Action:** {turn_data['action_name']}"
        if turn_data["action_name"] != "initial"
        else ""
    )
    st.markdown(f"**Turn {turn_data['turn_number']}**{action_text}")

    # Game state visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display ASCII art in a monospace font
        st.code(turn_data["room_text"], language=None)

    with col2:
        # Consolidated turn info
        turn_info = f"""
        **Boxes on Target:** {turn_data["boxes_on_target"]}  
        **Steps:** {turn_data["num_steps"]}  
        **Reward:** {turn_data["reward"]:.3f}  
        **Position:** ({turn_data["player_position"][0]}, {turn_data["player_position"][1]})
        """
        st.markdown(turn_info)

        # Status
        if turn_data["terminated"]:
            st.success("✅ Solved!")
        elif turn_data["truncated"]:
            st.warning("⏰ Timeout")
        else:
            st.info("🎯 Active")


def render_sokoban_turn_compact(turn_data: Dict):
    """Render a compact view of a Sokoban turn."""
    col1, col2, col3 = st.columns(3)

    with col1:
        # Mini ASCII display (first few lines)
        room_lines = turn_data["room_text"].split("\n")
        mini_room = (
            "\n".join(room_lines[:3]) + "..."
            if len(room_lines) > 3
            else turn_data["room_text"]
        )
        st.code(mini_room, language=None)

    with col2:
        st.write(f"**Action:** {turn_data['action_name']}")
        st.write(f"**Boxes on Target:** {turn_data['boxes_on_target']}")
        st.write(f"**Steps:** {turn_data['num_steps']}")

    with col3:
        st.write(f"**Reward:** {turn_data['reward']:.3f}")
        if turn_data["terminated"]:
            st.success("✅ Solved")
        elif turn_data["truncated"]:
            st.warning("⏰ Timeout")
        else:
            st.info("🎯 Active")


def render_minigrid_turn(turn_data: Dict):
    """Render a detailed view of a single MiniGrid turn."""
    st.subheader(f"Step {turn_data['turn_number']}")

    # Game state visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        # Images
        if turn_data["images"]:
            st.subheader("🎮 Grid State")
            # Show the grid visualization
            for i, img in enumerate(
                turn_data["images"][:1]
            ):  # Usually one image per step
                try:
                    st.image(img, caption=f"Grid state", width=400)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
        else:
            st.info("No grid visualization available for this step")

    with col2:
        # Mission
        if turn_data["mission"]:
            st.subheader("🎯 Mission")
            st.text(turn_data["mission"])

        # Action taken
        if turn_data["actions"]:
            st.subheader("🎮 Action")
            for action in turn_data["actions"]:
                st.info(f"{action['name'].upper()}")

        # Rewards
        if turn_data["rewards"]:
            st.subheader("🏆 Reward")
            for reward in turn_data["rewards"]:
                if reward != 0:
                    if reward > 0:
                        st.success(f"+{reward:.2f}")
                    else:
                        st.error(f"{reward:.2f}")
                else:
                    st.text("0.00")


def render_minigrid_turn_compact(turn_data: Dict):
    """Render a compact view of a MiniGrid turn."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if turn_data["actions"]:
            actions = [action["name"] for action in turn_data["actions"]]
            st.write(f"**Action:** {', '.join(actions)}")
        else:
            st.write("**Action:** None")

    with col2:
        if turn_data["rewards"]:
            non_zero_rewards = [r for r in turn_data["rewards"] if r != 0]
            if non_zero_rewards:
                st.write(f"**Reward:** {sum(non_zero_rewards):.2f}")
            else:
                st.write("**Reward:** 0.00")

    with col3:
        if turn_data["images"]:
            # Show thumbnail
            st.image(turn_data["images"][0], width=100)


def render_evaluation_summary(evaluations: List[Dict]):
    """Render evaluation summary table."""
    if not evaluations:
        st.info("No evaluations found")
        return

    df = pd.DataFrame(evaluations)
    st.dataframe(
        df[
            [
                "run_id",
                "timestamp",
                "num_trajectories",
                "success_rate",
                "avg_achievements",
            ]
        ],
        use_container_width=True,
    )


def render_trajectory_summary(trajectories: List[Dict]):
    """Render trajectory summary table."""
    if not trajectories:
        st.info("No trajectories found")
        return

    trajectory_summary = [
        {
            "Model": traj["model_name"],
            "Difficulty": traj["difficulty"],
            "Success": traj["success"],
            "Final Reward": traj["final_reward"],
            "Steps": traj["num_steps"],
            "Seed": traj["seed"],
        }
        for traj in trajectories
    ]

    df = pd.DataFrame(trajectory_summary)
    st.dataframe(df, use_container_width=True)


# Main app
def main():
    # Custom CSS for compact layout
    st.markdown(
        """
    <style>
    .main > div {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem 0;
    }
    .stMetric > div {
        font-size: 0.7rem !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        font-size: 0.7rem !important;
    }
    .stMetric [data-testid="metric-container"] label {
        font-size: 0.6rem !important;
    }
    .stMetric [data-testid="metric-container"] > div > div {
        font-size: 0.9rem !important;
        font-weight: 600;
    }
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0.5rem !important;
    }
    h3 {
        font-size: 1rem !important;
        margin-bottom: 0.2rem !important;
        margin-top: 0.3rem !important;
    }
    .stSelectbox > div > div {
        font-size: 0.8rem !important;
    }
    .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.25rem 0.5rem !important;
    }
    .stInfo {
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
    }
    .stWarning {
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
    }
    .stSuccess {
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
    }
    .stError {
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
    }
    div[data-testid="stSidebar"] {
        width: 280px !important;
    }
    div[data-testid="stSidebar"] > div {
        font-size: 0.7rem !important;
        padding: 0.2rem !important;
    }
    div[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.3rem !important;
    }
    div[data-testid="stSidebar"] .stSelectbox > div {
        margin-bottom: 0.1rem !important;
    }
    div[data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
    }
    div[data-testid="stSidebar"] .stMarkdown p {
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
        font-size: 0.65rem !important;
    }
    div[data-testid="stSidebar"] .stButton {
        margin-bottom: 0.2rem !important;
        margin-top: 0.2rem !important;
    }
    div[data-testid="stSidebar"] .stInfo {
        margin-bottom: 0.2rem !important;
        margin-top: 0.2rem !important;
        padding: 0.3rem !important;
    }
    div[data-testid="stSidebar"] h2 {
        font-size: 0.8rem !important;
        margin-bottom: 0.2rem !important;
        margin-top: 0.3rem !important;
    }
    div[data-testid="stSidebar"] h3 {
        font-size: 0.75rem !important;
        margin-bottom: 0.1rem !important;
        margin-top: 0.2rem !important;
    }
    div[data-testid="stSidebar"] hr {
        margin: 0.3rem 0 !important;
    }
    .stMarkdown {
        font-size: 0.8rem !important;
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
    }
    .stText {
        font-size: 0.8rem !important;
    }
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    .stExpander {
        margin-bottom: 0.2rem !important;
    }
    .stExpander > div {
        padding: 0.3rem !important;
    }
    .stSlider {
        margin-bottom: 0.3rem !important;
        margin-top: 0.1rem !important;
    }
    .stCode {
        font-size: 0.65rem !important;
        margin-bottom: 0.2rem !important;
    }
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🎮 Synth Trace Viewer")
    st.caption("Interactive viewer for game AI traces")

    # Get database connection
    conn = get_db_connection()
    if not conn:
        st.stop()

    # Sidebar for navigation
    st.sidebar.markdown("**Navigation**")

    # Database status and sync button in one line
    try:
        total_evals = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        total_trajs = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        st.sidebar.markdown(f"📊 **DB:** {total_evals} evals, {total_trajs} trajs")
    except:
        pass

    # Sync traces button
    if st.sidebar.button("🔄 Sync", help="Discover and import new evaluation runs"):
        with st.spinner("Syncing traces..."):
            sync_results = sync_traces(conn)

        # Show results in sidebar
        if isinstance(sync_results, list):
            for result in sync_results:
                if "✅" in result:
                    st.sidebar.success(result)
                elif "⚠️" in result:
                    st.sidebar.warning(result)
                else:
                    st.sidebar.error(result)
        else:
            st.sidebar.info(sync_results)

        # Rerun to refresh the data
        st.rerun()

        # Load environments
    environments = load_environments(conn)
    if not environments:
        st.error("No environments found in database")
        st.stop()

    env_names = [env["display_name"] for env in environments]
    selected_env_idx = st.sidebar.selectbox(
        "**Select Environment**",
        range(len(env_names)),
        format_func=lambda i: env_names[i],
    )
    selected_env = environments[selected_env_idx]

    if selected_env:
        # Load regular evaluations
        all_evaluations = load_evaluations(conn, selected_env["id"], archived=False)

        # Filter evaluations to only include those with trajectories that have agent reasoning
        evaluations = []
        for eval_data in all_evaluations:
            trajectories = load_trajectories(conn, eval_data["id"])
            if (
                trajectories
            ):  # Only include evaluations that have trajectories with agent reasoning
                evaluations.append(eval_data)

        # Load archived evaluations
        all_archived_evaluations = load_evaluations(
            conn, selected_env["id"], archived=True
        )

        # Filter archived evaluations to only include those with trajectories that have agent reasoning
        archived_evaluations = []
        for eval_data in all_archived_evaluations:
            trajectories = load_trajectories(conn, eval_data["id"])
            if (
                trajectories
            ):  # Only include evaluations that have trajectories with agent reasoning
                archived_evaluations.append(eval_data)

        # Check if we have any evaluations at all
        if not evaluations and not archived_evaluations:
            st.warning(
                f"No evaluations with agent reasoning found for {selected_env['display_name']}"
            )
            st.stop()

        # Show filtering info if some evaluations were filtered out
        total_evals = len(all_evaluations)
        filtered_evals = len(evaluations)
        total_archived = len(all_archived_evaluations)
        filtered_archived = len(archived_evaluations)

        if total_evals > filtered_evals or total_archived > filtered_archived:
            st.sidebar.markdown(
                f"ℹ️ **Filtered:** {filtered_evals}/{total_evals} evals, {filtered_archived}/{total_archived} archived with reasoning"
            )

        # Evaluation selector - choose between regular and archived
        selected_eval = None

        # Initialize session state for evaluation type
        if "evaluation_type" not in st.session_state:
            st.session_state.evaluation_type = "regular" if evaluations else "archived"

        # Type selector
        eval_type_options = []
        if evaluations:
            eval_type_options.append(("regular", f"Active ({len(evaluations)})"))
        if archived_evaluations:
            eval_type_options.append(
                ("archived", f"Archived ({len(archived_evaluations)})")
            )

        if len(eval_type_options) > 1:
            selected_type = st.sidebar.selectbox(
                "**Evaluation Type**",
                options=[opt[0] for opt in eval_type_options],
                format_func=lambda x: next(
                    opt[1] for opt in eval_type_options if opt[0] == x
                ),
                index=0
                if st.session_state.evaluation_type == eval_type_options[0][0]
                else 1,
            )
            st.session_state.evaluation_type = selected_type
        else:
            st.session_state.evaluation_type = (
                eval_type_options[0][0] if eval_type_options else "regular"
            )

        # Show appropriate evaluation selector
        if st.session_state.evaluation_type == "regular" and evaluations:
            eval_options = [
                f"Run {eval['run_id']} - {eval['num_trajectories']} trajs ({format_timestamp(eval['timestamp'])})"
                for eval in evaluations
            ]
            selected_eval_idx = st.sidebar.selectbox(
                "**Select Evaluation**",
                range(len(eval_options)),
                format_func=lambda i: eval_options[i],
            )
            selected_eval = evaluations[selected_eval_idx]
        elif st.session_state.evaluation_type == "archived" and archived_evaluations:
            archived_eval_options = [
                f"Run {eval['run_id']} - {eval['num_trajectories']} trajs ({format_timestamp(eval['timestamp'])})"
                for eval in archived_evaluations
            ]
            selected_archived_idx = st.sidebar.selectbox(
                "**Select Evaluation**",
                range(len(archived_eval_options)),
                format_func=lambda i: archived_eval_options[i],
            )
            selected_eval = archived_evaluations[selected_archived_idx]

        # Fallback if no evaluation selected
        if not selected_eval:
            if evaluations:
                selected_eval = evaluations[0]
            elif archived_evaluations:
                selected_eval = archived_evaluations[0]

        # Load trajectories for selected evaluation (used for both metrics and trajectory selection)
        trajectories = load_trajectories(conn, selected_eval["id"])

        # === SHOW RUN-LEVEL INFO IN SIDEBAR ===
        st.sidebar.markdown("**📊 Run Details**")

        # Compute actual metrics from trajectories
        if trajectories:
            avg_reward = sum(traj["final_reward"] for traj in trajectories) / len(
                trajectories
            )
            success_rate = sum(1 for traj in trajectories if traj["success"]) / len(
                trajectories
            )

            # Calculate average achievements (only for trajectories that have achievements)
            achievements_list = [
                len(traj["achievements"])
                for traj in trajectories
                if traj["achievements"]
            ]
            avg_achievements = (
                sum(achievements_list) / len(achievements_list)
                if achievements_list
                else 0
            )

            # Format metrics
            avg_reward_str = f"{avg_reward:.3f}"
            success_rate_str = f"{success_rate:.1%}"
            avg_achievements_str = (
                f"{avg_achievements:.1f}" if avg_achievements > 0 else "N/A"
            )
        else:
            avg_reward_str = "N/A"
            success_rate_str = "N/A"
            avg_achievements_str = "N/A"

        st.sidebar.markdown(
            f"**Trajs:** {len(trajectories)} | **Avg Reward:** {avg_reward_str} | **Success:** {success_rate_str} | **Avg Ach:** {avg_achievements_str}"
        )

        # Archive/Unarchive buttons
        is_archived = selected_eval.get("archived", False)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if not is_archived:
                if st.button("📁 Archive", help="Archive this evaluation"):
                    if archive_evaluation(conn, selected_eval["id"], True):
                        st.success("Evaluation archived!")
                        st.rerun()
            else:
                if st.button("📤 Unarchive", help="Unarchive this evaluation"):
                    if archive_evaluation(conn, selected_eval["id"], False):
                        st.success("Evaluation unarchived!")
                        st.rerun()

        # Get total count before filtering for comparison
        total_query = """
        SELECT COUNT(*) FROM trajectories WHERE eval_id = ?
        """
        total_count = conn.execute(total_query, [selected_eval["id"]]).fetchone()[0]
        filtered_count = len(trajectories)

        if not trajectories:
            if total_count > 0:
                st.warning(
                    f"No trajectories with agent reasoning found for selected evaluation (filtered out {total_count} incomplete traces)"
                )
            else:
                st.warning(f"No trajectories found for selected evaluation")
            st.stop()

        # Show filtering info if some were filtered out
        if total_count > filtered_count:
            st.sidebar.markdown(
                f"ℹ️ **Filtered:** {filtered_count}/{total_count} with reasoning"
            )

        # Show trajectory selector in sidebar
        trajectory_options = [
            f"{traj['model_name']} - {traj['difficulty']} (R: {traj['final_reward']:.3f})"
            for traj in trajectories
        ]
        selected_traj_idx = st.sidebar.selectbox(
            "**Select Trajectory**",
            range(len(trajectory_options)),
            format_func=lambda i: trajectory_options[i],
        )

        selected_trajectory = trajectories[selected_traj_idx]

        # === MAIN PANEL ===

        # Show selected trajectory details
        render_trajectory_details(selected_trajectory, conn)


if __name__ == "__main__":
    main()
