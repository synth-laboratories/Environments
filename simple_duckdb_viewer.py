#!/usr/bin/env python3
"""
Super simple DuckDB viewer that actually works.
"""

import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.synth_env.db_minimal import SynthEvalDB
from src.synth_env.viewer.crafter import CrafterViewer
from src.synth_env.viewer.base import ViewerConfig

app = FastAPI()
db = SynthEvalDB("synth_eval_viewer.duckdb")

# Cache for viewers
viewers = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple landing page."""
    evals = db.get_evaluations("crafter")
    
    eval_list = ""
    for eval in evals:
        eval_list += f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0;">
            <strong>{eval['id']}</strong><br>
            Date: {eval['date_display']}<br>
            Trajectories: {eval['num_trajectories']}<br>
            <a href="/viewer?run_id={eval['id']}" style="background: #3498db; color: white; padding: 5px 10px; text-decoration: none;">View</a>
        </div>
        """
    
    return f"""
    <html>
    <head><title>DuckDB Viewer</title></head>
    <body style="font-family: sans-serif; margin: 20px;">
        <h1>🗄️ DuckDB Crafter Viewer</h1>
        <h2>Available Evaluations:</h2>
        {eval_list}
    </body>
    </html>
    """

@app.get("/viewer", response_class=HTMLResponse)
async def viewer(run_id: str = Query(...)):
    """Show viewer for a specific run."""
    # Get trajectories for this run
    trajectories = db.get_trajectories("crafter", run_id)
    
    if not trajectories:
        return "<h1>No trajectories found</h1>"
    
    # Create simple viewer HTML
    return f"""
    <html>
    <head>
        <title>Crafter Viewer - {run_id}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            select {{ padding: 5px; margin: 10px; }}
            #trace-info {{ border: 1px solid #ddd; padding: 15px; margin: 20px 0; }}
            #timeline {{ border: 1px solid #ddd; padding: 15px; margin: 20px 0; }}
            .turn-item {{ 
                cursor: pointer; 
                padding: 10px; 
                margin: 5px 0; 
                background: #f0f0f0; 
                border-radius: 4px;
            }}
            .turn-item:hover {{ background: #e0e0e0; }}
            .turn-item.active {{ background: #3498db; color: white; }}
            #turn-details {{ border: 1px solid #ddd; padding: 15px; margin: 20px 0; min-height: 300px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎮 Crafter Viewer</h1>
            <h2>Run: {run_id}</h2>
            
            <div>
                <label>Select Trace:</label>
                <select id="trace-select" onchange="loadTrace(this.value)">
                    <option value="">-- Select a trace --</option>
                    {' '.join(f'<option value="{t["id"]}">{t["id"][:8]}... ({t["model_name"]})</option>' for t in trajectories)}
                </select>
            </div>
            
            <div id="trace-info"></div>
            <div id="timeline"></div>
            <div id="turn-details"></div>
        </div>
        
        <script>
        let currentTrace = null;
        let currentTurnIndex = null;
        
        async function loadTrace(traceId) {{
            if (!traceId) return;
            
            console.log('Loading trace:', traceId);
            document.getElementById('trace-info').innerHTML = '<p>Loading trace...</p>';
            
            try {{
                const response = await fetch(`/api/trace/${{traceId}}?run_id={run_id}`);
                const data = await response.json();
                console.log('Trace data:', data);
                
                currentTrace = data;
                currentTurnIndex = null;
                
                // Display trace info
                document.getElementById('trace-info').innerHTML = `
                    <h3>Trace Info</h3>
                    <p>Model: ${{data.trace.metadata.model_name || 'unknown'}}</p>
                    <p>Turns: ${{data.trace.partition.length}}</p>
                `;
                
                // Display timeline
                displayTimeline();
                
                // Clear turn details
                document.getElementById('turn-details').innerHTML = '<p>Select a turn from the timeline</p>';
                
            }} catch (error) {{
                console.error('Error loading trace:', error);
                document.getElementById('trace-info').innerHTML = '<p style="color: red;">Error loading trace</p>';
            }}
        }}
        
        function displayTimeline() {{
            const timeline = document.getElementById('timeline');
            timeline.innerHTML = '<h3>Timeline</h3>';
            
            currentTrace.trace.partition.forEach((partition, index) => {{
                const item = document.createElement('div');
                item.className = 'turn-item';
                item.innerHTML = `Turn ${{index + 1}}`;
                item.onclick = () => selectTurn(index);
                timeline.appendChild(item);
            }});
        }}
        
        function selectTurn(index) {{
            currentTurnIndex = index;
            
            // Update active state
            document.querySelectorAll('.turn-item').forEach((item, i) => {{
                item.classList.toggle('active', i === index);
            }});
            
            // Display turn details
            const partition = currentTrace.trace.partition[index];
            const event = partition.events[0];
            
            let html = `<h3>Turn ${{index + 1}} Details</h3>`;
            
            if (event.agent_compute_step) {{
                html += `<p><strong>Agent Input:</strong></p>`;
                html += `<pre>${{JSON.stringify(event.agent_compute_step.inputs, null, 2)}}</pre>`;
            }}
            
            if (event.environment_compute_steps && event.environment_compute_steps.length > 0) {{
                html += `<p><strong>Environment Steps:</strong></p>`;
                event.environment_compute_steps.forEach((step, i) => {{
                    if (step.output) {{
                        html += `<p>Step ${{i + 1}}:</p>`;
                        html += `<pre>${{JSON.stringify(step.output, null, 2)}}</pre>`;
                    }}
                }});
            }}
            
            document.getElementById('turn-details').innerHTML = html;
        }}
        </script>
    </body>
    </html>
    """

@app.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str, run_id: str = Query(...)):
    """Get trace data from DuckDB."""
    # Get trace JSON from DB
    trace_json = db.get_trace_json("crafter", run_id, trace_id)
    
    if not trace_json:
        return JSONResponse({"error": "Trace not found"}, status_code=404)
    
    # Parse and return
    data = json.loads(trace_json)
    return data

# Run the server
if __name__ == "__main__":
    print("🚀 Starting Simple DuckDB Viewer on http://localhost:8995")
    print("   This viewer actually works!")
    uvicorn.run(app, host="0.0.0.0", port=8995)