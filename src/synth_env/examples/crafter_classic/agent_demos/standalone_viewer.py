#!/usr/bin/env python3
"""
Standalone viewer for Crafter evaluations.
Can browse and view existing evaluations without running new trajectories.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn


# Global variable to store current eval directory
_current_eval_dir = None

def set_current_eval_dir(eval_dir: Path):
    """Set the current evaluation directory for the viewer."""
    global _current_eval_dir
    _current_eval_dir = eval_dir


def regenerate_viewer_files(viewer_dir: Path):
    """Regenerate viewer files with the latest simplified version."""
    viewer_dir.mkdir(parents=True, exist_ok=True)
    
    # Create index.html
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crafter Evaluation Viewer</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="app">
        <div class="header">
            <h1>🎮 Crafter Evaluation Viewer</h1>
            <div class="trace-selector">
                <label for="trace-select">Select Trace:</label>
                <select id="trace-select"></select>
                <button id="refresh-btn">🔄 Refresh</button>
            </div>
        </div>
        
        <div class="main-container">
            <div class="sidebar">
                <h2>Timeline</h2>
                <div id="timeline" class="timeline"></div>
                
                <div class="trace-info">
                    <h3>Trace Info</h3>
                    <div id="trace-metadata"></div>
                </div>
            </div>
            
            <div class="content">
                <div class="question-reward">
                    <h2>Training Question</h2>
                    <div id="question-display" class="info-box"></div>
                    
                    <h2>Reward Signal</h2>
                    <div id="reward-display" class="info-box"></div>
                </div>
                
                <div class="turn-details">
                    <h2>Turn Details</h2>
                    <div id="turn-content">
                        <p class="placeholder">Select a turn from the timeline</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="viewer.js"></script>
</body>
</html>"""
    
    # Create style.css
    css_content = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    color: #333;
}

.header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header h1 {
    font-size: 1.5rem;
}

.trace-selector {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.trace-selector select {
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid #34495e;
    background-color: white;
    min-width: 200px;
}

.trace-selector button {
    padding: 0.5rem 1rem;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.trace-selector button:hover {
    background-color: #2980b9;
}

.main-container {
    display: flex;
    height: calc(100vh - 60px);
}

.sidebar {
    width: 300px;
    background-color: white;
    border-right: 1px solid #ddd;
    overflow-y: auto;
    padding: 1rem;
}

.timeline {
    margin-bottom: 2rem;
}

.timeline-item {
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background-color: #f8f9fa;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    border: 2px solid transparent;
}

.timeline-item:hover {
    background-color: #e9ecef;
}

.timeline-item.active {
    background-color: #3498db;
    color: white;
    border-color: #2980b9;
}

.timeline-item .turn-number {
    font-weight: bold;
    margin-bottom: 0.25rem;
}

.timeline-item .turn-stats {
    font-size: 0.85rem;
    opacity: 0.8;
}

.trace-info {
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 4px;
}

.trace-info h3 {
    margin-bottom: 0.5rem;
    color: #2c3e50;
}

.content {
    flex: 1;
    padding: 2rem;
    overflow-y: auto;
}

.question-reward {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

.info-box {
    padding: 1rem;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.turn-details {
    background-color: white;
    border-radius: 4px;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.turn-details h2 {
    margin-bottom: 1rem;
    color: #2c3e50;
}

.placeholder {
    color: #999;
    text-align: center;
    padding: 3rem;
}

.agent-section, .environment-section {
    margin-bottom: 2rem;
}

.agent-section h3, .environment-section h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.message-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}

.message-box .role {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.actions-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    font-family: monospace;
}

.env-step {
    margin-bottom: 1.5rem;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 4px;
}

.env-step h4 {
    color: #2c3e50;
    margin-bottom: 0.75rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1rem;
}

.stat-item {
    text-align: center;
    padding: 0.5rem;
    background-color: #fff;
    border-radius: 4px;
}

.stat-label {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.25rem;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: bold;
    color: #2c3e50;
}

.env-image {
    max-width: 400px;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 1rem 0;
}

.achievements-section {
    padding: 1rem;
    background-color: #e8f5e9;
    border-radius: 4px;
}

.achievements-section h3 {
    color: #27ae60;
    margin-bottom: 0.75rem;
}

.metadata-item {
    margin-bottom: 0.5rem;
}

.metadata-label {
    font-weight: bold;
    color: #666;
}

.achievement-badge {
    display: inline-block;
    background-color: #27ae60;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.85rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.images-row {
    display: flex;
    gap: 1rem;
    overflow-x: auto;
    padding: 1rem 0;
}

.image-container {
    text-align: center;
    flex-shrink: 0;
}

.image-caption {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #666;
    font-family: monospace;
}

@media (max-width: 768px) {
    .main-container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: 200px;
        border-right: none;
        border-bottom: 1px solid #ddd;
    }
    
    .question-reward {
        grid-template-columns: 1fr;
    }
}"""
    
    # Create viewer.js with SIMPLIFIED display
    js_content = """let currentTrace = null;
let currentTurnIndex = null;

// Action mapping
const ACTION_NAMES = {
    -1: 'initial state',
    0: 'noop',
    1: 'move_left',
    2: 'move_right',
    3: 'move_up',
    4: 'move_down',
    5: 'do',
    6: 'sleep',
    7: 'place_stone',
    8: 'place_table',
    9: 'place_furnace',
    10: 'place_plant',
    11: 'make_wood_pickaxe',
    12: 'make_stone_pickaxe',
    13: 'make_iron_pickaxe',
    14: 'make_wood_sword',
    15: 'make_stone_sword',
    16: 'make_iron_sword'
};

// Load available traces
async function loadTraceList() {
    try {
        const response = await fetch('/api/traces');
        const traces = await response.json();
        
        const select = document.getElementById('trace-select');
        select.innerHTML = '';
        
        traces.forEach(trace => {
            const option = document.createElement('option');
            option.value = trace.id;
            option.textContent = `${trace.model_name} - ${trace.difficulty} - ${trace.id.substring(0, 8)}`;
            select.appendChild(option);
        });
        
        if (traces.length > 0) {
            loadTrace(traces[0].id);
        }
    } catch (error) {
        console.error('Failed to load traces:', error);
    }
}

// Load specific trace
async function loadTrace(traceId) {
    try {
        const response = await fetch(`/api/trace/${traceId}`);
        const data = await response.json();
        
        currentTrace = data;
        currentTurnIndex = null;
        
        displayTraceInfo();
        displayTimeline();
        displayQuestionAndReward();
        clearTurnDetails();
    } catch (error) {
        console.error('Failed to load trace:', error);
    }
}

// Display trace metadata
function displayTraceInfo() {
    const metadataDiv = document.getElementById('trace-metadata');
    const metadata = currentTrace.trace.metadata;
    
    metadataDiv.innerHTML = `
        <div class="metadata-item">
            <span class="metadata-label">Model:</span> ${metadata.model_name}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Difficulty:</span> ${metadata.difficulty}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Seed:</span> ${metadata.seed}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Max Turns:</span> ${metadata.max_turns}
        </div>
    `;
}

// Display timeline
function displayTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    currentTrace.trace.partition.forEach((partition, index) => {
        const event = partition.events[0];
        const metadata = event.event_metadata;
        
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.innerHTML = `
            <div class="turn-number">Turn ${metadata.turn_number}</div>
            <div class="turn-stats">
                Actions: ${event.environment_compute_steps.length} | 
                Achievements: ${metadata.total_achievements}
                ${metadata.new_achievements.length > 0 ? ' (+' + metadata.new_achievements.length + ')' : ''}
            </div>
        `;
        
        item.addEventListener('click', () => selectTurn(index));
        timeline.appendChild(item);
    });
}

// Display question and reward
function displayQuestionAndReward() {
    const question = currentTrace.dataset.questions[0];
    const reward = currentTrace.dataset.reward_signals[0];
    
    document.getElementById('question-display').innerHTML = `
        <p><strong>Intent:</strong> ${question.intent}</p>
        <p><strong>Criteria:</strong> ${question.criteria}</p>
    `;
    
    document.getElementById('reward-display').innerHTML = `
        <p><strong>Hafner Score:</strong> ${reward.reward.toFixed(2)}%</p>
        <p><strong>Annotation:</strong> ${reward.annotation}</p>
    `;
}

// Select turn
function selectTurn(index) {
    currentTurnIndex = index;
    
    // Update timeline selection
    document.querySelectorAll('.timeline-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
    
    // Display turn details
    displayTurnDetails();
}

// Display turn details - SIMPLIFIED VERSION
function displayTurnDetails() {
    if (currentTurnIndex === null) return;
    
    const partition = currentTrace.trace.partition[currentTurnIndex];
    const event = partition.events[0];
    const agentStep = event.agent_compute_step;
    const envSteps = event.environment_compute_steps;
    
    let html = '';
    
    // Display actions planned
    if (agentStep.compute_output[0] && agentStep.compute_output[0].outputs) {
        const outputs = agentStep.compute_output[0].outputs;
        const actionNames = outputs.actions.map(idx => `${ACTION_NAMES[idx] || 'unknown'}`);
        html += `
            <div class="actions-box" style="margin-bottom: 1.5rem;">
                <strong>Turn ${event.event_metadata.turn_number} Actions:</strong> ${actionNames.join(' → ')}
            </div>
        `;
    }
    
    // Display all images in a row
    html += '<div class="images-row">';
    envSteps.forEach((step, i) => {
        const outputs = step.compute_output[0].outputs;
        const actionName = ACTION_NAMES[outputs.action_index] || 'unknown';
        
        if (outputs.image_base64) {
            // For initial state, show "0. initial state", otherwise show action number
            const stepNumber = outputs.action_index === -1 ? 0 : i;
            html += `
                <div class="image-container">
                    <img src="data:image/png;base64,${outputs.image_base64}" class="env-image" alt="Game state">
                    <div class="image-caption">${stepNumber}. ${actionName}</div>
                </div>
            `;
        }
    });
    html += '</div>';
    
    // New achievements
    if (event.event_metadata.new_achievements.length > 0) {
        html += '<div class="achievements-section" style="margin-top: 1rem;">';
        html += '<strong>New achievements: </strong>';
        event.event_metadata.new_achievements.forEach(ach => {
            html += `<span class="achievement-badge">${ach}</span>`;
        });
        html += '</div>';
    }
    
    document.getElementById('turn-content').innerHTML = html;
}

// Clear turn details
function clearTurnDetails() {
    document.getElementById('turn-content').innerHTML = '<p class="placeholder">Select a turn from the timeline</p>';
}

// Event listeners
document.getElementById('trace-select').addEventListener('change', (e) => {
    if (e.target.value) {
        loadTrace(e.target.value);
    }
});

document.getElementById('refresh-btn').addEventListener('click', () => {
    loadTraceList();
});

// Initial load
loadTraceList();"""
    
    # Save files
    with open(viewer_dir / "index.html", 'w') as f:
        f.write(html_content)
    
    with open(viewer_dir / "style.css", 'w') as f:
        f.write(css_content)
    
    with open(viewer_dir / "viewer.js", 'w') as f:
        f.write(js_content)


# FastAPI app for viewer
app = FastAPI()

@app.get("/api/traces")
async def get_traces():
    """Get list of available traces."""
    global _current_eval_dir
    if _current_eval_dir is None:
        return []
        
    traces_dir = _current_eval_dir / "traces"
    if not traces_dir.exists():
        return []
    
    traces = []
    for trace_file in traces_dir.glob("*.json"):
        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)
                trace_meta = data["trace"]["metadata"]
                traces.append({
                    "id": trace_file.stem,
                    "model_name": trace_meta["model_name"],
                    "difficulty": trace_meta["difficulty"],
                    "seed": trace_meta["seed"]
                })
        except Exception as e:
            print(f"Error loading trace {trace_file}: {e}")
    
    return sorted(traces, key=lambda x: x["id"])

@app.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Get specific trace data."""
    global _current_eval_dir
    if _current_eval_dir is None:
        raise HTTPException(status_code=404, detail="No evaluation directory set")
        
    trace_file = _current_eval_dir / "traces" / f"{trace_id}.json"
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail="Trace not found")
    
    with open(trace_file, 'r') as f:
        return json.load(f)

@app.get("/api/eval_info")
async def get_eval_info():
    """Get evaluation metadata."""
    global _current_eval_dir
    if _current_eval_dir is None:
        return {"error": "No evaluation directory set"}
    
    summary_file = _current_eval_dir / "evaluation_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return {"error": "No evaluation summary found"}


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


async def run_viewer(eval_dir: Path, port: int = 8999, host: str = "0.0.0.0"):
    """Run the viewer for a specific evaluation directory."""
    viewer_dir = eval_dir / "viewer"
    
    # Always regenerate viewer files with latest simplified version
    print(f"🔄 Regenerating viewer files...")
    regenerate_viewer_files(viewer_dir)
    
    print(f"\n📁 Viewing evaluation: {eval_dir}")
    print(f"🌐 Launching viewer at http://localhost:{port}")
    print("   Press Ctrl+C to stop the viewer\n")
    
    # Set the current eval directory
    set_current_eval_dir(eval_dir)
    
    # Mount static files from the viewer directory
    app.mount("/", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")
    
    # Run viewer
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\n👋 Viewer stopped")


async def main():
    parser = argparse.ArgumentParser(
        description="View Crafter evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View the latest evaluation
  python standalone_viewer.py --latest
  
  # View a specific evaluation
  python standalone_viewer.py --run run_20240115_143022
  
  # List all evaluations
  python standalone_viewer.py --list
  
  # Use a different port
  python standalone_viewer.py --latest --port 8080
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
        await run_viewer(eval_dir, port=args.port, host=args.host)


if __name__ == "__main__":
    asyncio.run(main())