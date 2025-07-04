#!/usr/bin/env python3
"""
Standalone viewer for Crafter evaluations using shared viewer components.
"""

import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.synth_env.viewer import ViewerConfig
from src.synth_env.viewer.crafter import CrafterViewer
from src.synth_env.viewer.server import ViewerServer


def find_latest_evaluation(base_dir: Path = Path("src/evals/crafter")) -> Optional[Path]:
    """Find the most recent evaluation directory."""
    if not base_dir.exists():
        return None
    
    run_dirs = list(base_dir.glob("run_*"))
    if not run_dirs:
        return None
    
    # Sort by directory name (timestamp)
    run_dirs.sort(reverse=True)
    return run_dirs[0]


def list_evaluations(base_dir: Path = Path("src/evals/crafter")) -> list:
    """List all available evaluations."""
    if not base_dir.exists():
        return []
    
    evaluations = []
    for run_dir in sorted(base_dir.glob("run_*"), reverse=True):
        if run_dir.is_dir():
            summary_file = run_dir / "evaluation_summary.json"
            if summary_file.exists():
                import json
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    evaluations.append({
                        "path": run_dir,
                        "name": run_dir.name,
                        "timestamp": summary["evaluation_metadata"]["timestamp"],
                        "models": summary.get("models_evaluated", []),
                        "difficulties": summary.get("difficulties_evaluated", []),
                        "num_trajectories": summary["evaluation_metadata"]["num_trajectories"]
                    })
    return evaluations


async def main():
    parser = argparse.ArgumentParser(
        description="View Crafter evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View the latest evaluation
  python standalone_viewer_v2.py --latest
  
  # View a specific evaluation
  python standalone_viewer_v2.py --run run_20240115_143022
  
  # List all evaluations
  python standalone_viewer_v2.py --list
  
  # Use a different port
  python standalone_viewer_v2.py --latest --port 8080
        """
    )
    
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="src/evals/crafter",
        help="Base directory for evaluations (default: src/evals/crafter)"
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Specific run ID to view (e.g., run_20240115_143022)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="View the latest evaluation"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available evaluations and exit"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8999,
        help="Port to run viewer on (default: 8999)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    base_dir = Path(args.eval_dir)
    
    # List mode
    if args.list:
        evaluations = list_evaluations(base_dir)
        if not evaluations:
            print(f"No evaluations found in {base_dir}")
            return
        
        print(f"\n📊 Available Crafter Evaluations in {base_dir}:\n")
        for eval_info in evaluations:
            try:
                ts = datetime.fromisoformat(eval_info["timestamp"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = eval_info["timestamp"]
            
            print(f"  {eval_info['name']}")
            print(f"    Time: {ts_str}")
            print(f"    Models: {', '.join(eval_info['models'])}")
            print(f"    Difficulties: {', '.join(eval_info['difficulties'])}")
            print(f"    Trajectories: {eval_info['num_trajectories']}")
            print()
        return
    
    # Determine which evaluation to view
    eval_dir = None
    
    if args.run:
        # Specific run requested
        eval_dir = base_dir / args.run
        if not eval_dir.exists():
            print(f"❌ Evaluation not found: {eval_dir}")
            return
    elif args.latest:
        # Latest evaluation requested
        eval_dir = find_latest_evaluation(base_dir)
        if not eval_dir:
            print(f"❌ No evaluations found in {base_dir}")
            return
    else:
        # Interactive selection
        evaluations = list_evaluations(base_dir)
        if not evaluations:
            print(f"No evaluations found in {base_dir}")
            return
        
        print(f"\n📊 Available Crafter Evaluations:\n")
        for i, eval_info in enumerate(evaluations):
            try:
                ts = datetime.fromisoformat(eval_info["timestamp"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = eval_info["timestamp"]
            
            print(f"{i+1}. {eval_info['name']} - {ts_str}")
            print(f"   Models: {', '.join(eval_info['models'])}")
            print(f"   Difficulties: {', '.join(eval_info['difficulties'])}")
        
        print("\nEnter the number of the evaluation to view (or 'q' to quit): ", end='')
        choice = input().strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(evaluations):
                eval_dir = evaluations[idx]["path"]
            else:
                print("Invalid selection")
                return
        except ValueError:
            print("Invalid input")
            return
    
    # Run the viewer
    if eval_dir:
        config = ViewerConfig(
            eval_dir=eval_dir,
            port=args.port,
            host=args.host
        )
        viewer = CrafterViewer(config)
        server = ViewerServer(viewer)
        await server.run()


if __name__ == "__main__":
    asyncio.run(main())