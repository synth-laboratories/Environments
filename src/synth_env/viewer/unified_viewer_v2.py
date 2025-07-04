#!/usr/bin/env python3
"""
Unified Viewer V2 - Plugin-based architecture with environment-specific visualizations.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from .base_v2 import viewer_registry
from .crafter_v2 import CrafterViewer  # This registers itself


class UnifiedViewerV2:
    """Unified viewer with plugin support for different environments."""
    
    def __init__(self, eval_dir: Path):
        # ASSERTION: eval_dir must exist
        assert eval_dir.exists(), f"Evaluation directory does not exist: {eval_dir}"
        self.eval_dir = eval_dir
        
        # ASSERTION: traces directory must exist
        self.traces_dir = eval_dir / "traces"
        assert self.traces_dir.exists(), f"Traces directory does not exist: {self.traces_dir}"
        
        # Detect environment
        self.env_name = viewer_registry.detect_environment(eval_dir)
        if not self.env_name:
            # Try to detect from any trace file
            trace_files = list(self.traces_dir.glob("*.json"))
            if trace_files:
                # Look at first trace to guess environment
                with open(trace_files[0], 'r') as f:
                    sample_data = json.load(f)
                    # Simple heuristics
                    if 'crafter' in str(trace_files[0]).lower():
                        self.env_name = 'crafter'
                    # Add more heuristics as needed
        
        print(f"🔍 Detected environment: {self.env_name or 'unknown'}")
        
        # Get environment viewer
        self.env_viewer = viewer_registry.get(self.env_name) if self.env_name else None
        if self.env_viewer:
            print(f"✅ Using {self.env_name} viewer plugin")
        else:
            print(f"⚠️  No viewer plugin found, using generic viewer")
        
        self.app = FastAPI()
        self.traces: Dict[str, Any] = {}
        self._setup_routes()
    
    def load_traces(self) -> Dict[str, Any]:
        """Load all trace files with validation."""
        self.traces.clear()
        trace_files = list(self.traces_dir.glob("*.json"))
        
        print(f"🔍 Found {len(trace_files)} trace files in {self.traces_dir}")
        
        errors = []
        for trace_file in trace_files:
            try:
                print(f"  Loading {trace_file.name}...")
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                
                # Basic validation
                assert isinstance(data, dict), f"Trace {trace_file} is not a dict"
                
                # Environment-specific validation if available
                if self.env_viewer:
                    try:
                        self.env_viewer.validate_trace_structure(data)
                        print(f"    ✅ Valid {self.env_name} trace")
                    except AssertionError as e:
                        print(f"    ⚠️  Validation warning: {e}")
                        # Continue anyway - we want to see the data
                
                trace_id = trace_file.stem
                self.traces[trace_id] = data
                
            except Exception as e:
                error_msg = f"Failed to load {trace_file.name}: {str(e)}"
                print(f"    ❌ {error_msg}")
                errors.append(error_msg)
                traceback.print_exc()
        
        print(f"\n📊 Loaded {len(self.traces)} traces successfully")
        if errors:
            print(f"❌ {len(errors)} errors occurred")
            
        return {"loaded": len(self.traces), "errors": errors}
    
    def _setup_routes(self):
        """Set up all routes."""
        
        # Store instance reference for route handlers
        viewer_instance = self
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main viewer page with environment-specific features."""
            # Get custom CSS/JS if available
            custom_css = viewer_instance.env_viewer.get_custom_css() if viewer_instance.env_viewer else ""
            custom_js = viewer_instance.env_viewer.get_custom_javascript() if viewer_instance.env_viewer else ""
            
            env_title = viewer_instance.env_name.title() if viewer_instance.env_name else "Unknown"
            
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>Synth {env_title} Viewer</title>
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .subtitle {{
            opacity: 0.8;
            font-size: 14px;
            margin-top: 5px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .main-content {{
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }}
        .sidebar {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }}
        .content {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }}
        button {{ 
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }}
        button:hover {{ 
            background: #2980b9;
        }}
        select {{ 
            background: white;
            color: #333;
            border: 1px solid #ddd;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            width: 100%;
            margin: 10px 0;
        }}
        .error {{ 
            color: #e74c3c; 
            background: #fee; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px;
            border-left: 4px solid #e74c3c;
        }}
        .success {{ 
            color: #27ae60; 
            background: #efe; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px;
            border-left: 4px solid #27ae60;
        }}
        .info {{ 
            color: #3498db; 
            background: #e3f2fd; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }}
        .timeline {{
            margin-top: 20px;
        }}
        .timeline-item {{
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }}
        .timeline-item:hover {{
            background: #e9ecef;
        }}
        .timeline-item.active {{
            background: #3498db;
            color: white;
            border-color: #2980b9;
        }}
        .trace-summary {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .trace-summary h4 {{
            margin-top: 0;
        }}
        #status {{
            margin-bottom: 20px;
        }}
        {custom_css}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Synth {env_title} Viewer</h1>
        <div class="subtitle">Environment: {str(viewer_instance.eval_dir)}</div>
    </div>
    
    <div class="container">
        <div id="status"></div>
        
        <div class="controls">
            <button onclick="loadTraces()">Load Traces</button>
            <select id="trace-select" onchange="selectTrace(this.value)">
                <option value="">-- No traces loaded --</option>
            </select>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                <h3>Trace Info</h3>
                <div id="trace-info"></div>
                
                <h3>Timeline</h3>
                <div id="timeline" class="timeline"></div>
            </div>
            
            <div class="content">
                <div id="turn-content">
                    <p style="text-align: center; color: #999; padding: 40px;">
                        Load a trace and select a turn to view details
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    let traces = {{}};
    let currentTrace = null;
    let currentTurnIndex = null;
    
    function showStatus(message, type = 'info') {{
        const status = document.getElementById('status');
        status.className = type;
        status.innerHTML = message;
    }}
    
    async function loadTraces() {{
        showStatus('Loading traces...', 'info');
        
        try {{
            const response = await fetch('/api/load-traces');
            const result = await response.json();
            
            if (result.error) {{
                showStatus(`Error: ${{result.error}}`, 'error');
                return;
            }}
            
            // Get trace list
            const listResponse = await fetch('/api/traces');
            traces = await listResponse.json();
            
            // Update dropdown
            const select = document.getElementById('trace-select');
            select.innerHTML = '<option value="">-- Select a trace --</option>';
            
            for (const [traceId, info] of Object.entries(traces)) {{
                const option = document.createElement('option');
                option.value = traceId;
                option.textContent = `${{traceId.substring(0, 8)}}... (${{info.model_name}}, ${{info.num_turns}} turns)`;
                select.appendChild(option);
            }}
            
            showStatus(`Loaded ${{Object.keys(traces).length}} traces`, 'success');
            
        }} catch (error) {{
            showStatus(`Failed to load traces: ${{error.message}}`, 'error');
        }}
    }}
    
    async function selectTrace(traceId) {{
        if (!traceId) {{
            document.getElementById('trace-info').innerHTML = '';
            document.getElementById('timeline').innerHTML = '';
            document.getElementById('turn-content').innerHTML = '';
            currentTraceId = null;
            return;
        }}
        
        currentTraceId = traceId;
        showStatus(`Loading trace ${{traceId}}...`, 'info');
        
        try {{
            const response = await fetch(`/api/trace/${{traceId}}`);
            currentTrace = await response.json();
            currentTurnIndex = null;
            
            // Display trace info
            const info = traces[traceId];
            let infoHtml = '<div class="trace-summary">';
            infoHtml += `<h4>Summary</h4>`;
            infoHtml += `<strong>Model:</strong> ${{info.model_name}}<br>`;
            infoHtml += `<strong>Turns:</strong> ${{info.num_turns}}<br>`;
            infoHtml += `<strong>Reward:</strong> ${{info.total_reward.toFixed(3)}}<br>`;
            
            if (info.custom_stats) {{
                for (const [key, value] of Object.entries(info.custom_stats)) {{
                    infoHtml += `<strong>${{key}}:</strong> ${{JSON.stringify(value)}}<br>`;
                }}
            }}
            infoHtml += '</div>';
            document.getElementById('trace-info').innerHTML = infoHtml;
            
            // Display timeline
            displayTimeline();
            
            showStatus('Trace loaded', 'success');
            
        }} catch (error) {{
            showStatus(`Failed to load trace: ${{error.message}}`, 'error');
        }}
    }}
    
    function displayTimeline() {{
        const timeline = document.getElementById('timeline');
        timeline.innerHTML = '';
        
        const numTurns = currentTrace.trace.partition.length;
        for (let i = 0; i < numTurns; i++) {{
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.textContent = `Turn ${{i + 1}}`;
            item.onclick = () => selectTurn(i);
            timeline.appendChild(item);
        }}
    }}
    
    async function selectTurn(index) {{
        currentTurnIndex = index;
        
        // Update timeline selection
        document.querySelectorAll('.timeline-item').forEach((item, i) => {{
            item.classList.toggle('active', i === index);
        }});
        
        // Get turn HTML from server
        try {{
            const response = await fetch(`/api/turn/${{currentTraceId}}/${{index}}`);
            const html = await response.text();
            document.getElementById('turn-content').innerHTML = html;
        }} catch (error) {{
            document.getElementById('turn-content').innerHTML = `
                <div class="error">Failed to render turn: ${{error.message}}</div>
            `;
        }}
    }}
    
    // Track current trace ID
    let currentTraceId = null;
    
    {custom_js}
    
    // Auto-load traces on page load
    window.onload = () => {{
        loadTraces();
    }};
    </script>
</body>
</html>"""
        
        @self.app.get("/api/load-traces")
        async def load_traces_endpoint():
            """Load traces."""
            try:
                result = viewer_instance.load_traces()
                return JSONResponse(result)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        
        @self.app.get("/api/traces")
        async def get_trace_list():
            """Get list of loaded traces with summaries."""
            trace_list = {}
            for trace_id, data in viewer_instance.traces.items():
                try:
                    if viewer_instance.env_viewer:
                        # Use environment-specific summary
                        summary = viewer_instance.env_viewer.get_trace_summary(data)
                        trace_list[trace_id] = summary
                    else:
                        # Generic summary
                        trace_list[trace_id] = {
                            'num_turns': len(data.get('trace', {}).get('partition', [])),
                            'model_name': 'unknown',
                            'total_reward': 0.0,
                            'custom_stats': {}
                        }
                except Exception as e:
                    trace_list[trace_id] = {"error": str(e)}
            
            return JSONResponse(trace_list)
        
        @self.app.get("/api/trace/{trace_id}")
        async def get_trace(trace_id: str):
            """Get specific trace data."""
            if trace_id not in viewer_instance.traces:
                raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
            
            return JSONResponse(viewer_instance.traces[trace_id])
        
        @self.app.get("/api/turn/{trace_id}/{turn_index}")
        async def get_turn_html(trace_id: str, turn_index: int):
            """Get HTML for a specific turn."""
            if trace_id not in viewer_instance.traces:
                raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
            
            trace_data = viewer_instance.traces[trace_id]
            
            if viewer_instance.env_viewer:
                # Use environment-specific rendering
                html = viewer_instance.env_viewer.render_turn_html(trace_data, turn_index)
            else:
                # Generic rendering
                html = f"""
                <div class="info">
                    <h3>Turn {turn_index + 1} (Generic View)</h3>
                    <p>No environment-specific viewer available.</p>
                    <pre>{json.dumps(trace_data['trace']['partition'][turn_index], indent=2)[:2000]}...</pre>
                </div>
                """
            
            return HTMLResponse(content=html)
        
        @self.app.get("/favicon.ico")
        async def favicon():
            """Serve the Synth favicon."""
            favicon_path = Path(__file__).parent.parent.parent.parent / "favicon_synth.ico"
            if favicon_path.exists():
                with open(favicon_path, 'rb') as f:
                    return Response(content=f.read(), media_type="image/x-icon")
            else:
                raise HTTPException(status_code=404, detail="Favicon not found")


async def run_unified_viewer(eval_dir: Path, port: int = 9001):
    """Run the unified viewer."""
    # Kill any existing process on the port
    import subprocess
    import platform
    
    if platform.system() == "Darwin":  # macOS
        # Find and kill process using the port
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], 
                capture_output=True, 
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(["kill", "-9", pid])
                print(f"✓ Killed existing process(es) on port {port}")
        except Exception:
            pass
    else:  # Linux
        try:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"], 
                capture_output=True
            )
        except Exception:
            pass
    
    print("\n" + "="*60)
    print("🔬 Synth Unified Viewer V2")
    print("="*60)
    print(f"Evaluation directory: {eval_dir}")
    print(f"Port: {port}")
    print(f"Available environments: {viewer_registry.list_environments()}")
    print("="*60 + "\n")
    
    try:
        viewer = UnifiedViewerV2(eval_dir)
        
        config = uvicorn.Config(
            viewer.app, 
            host="0.0.0.0", 
            port=port, 
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        print(f"\n🚀 Launching at http://localhost:{port}")
        print("   Press Ctrl+C to stop\n")
        
        await server.serve()
        
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Viewer V2")
    parser.add_argument("eval_dir", type=Path, help="Evaluation directory")
    parser.add_argument("--port", type=int, default=9001, help="Port to run on")
    
    args = parser.parse_args()
    
    asyncio.run(run_unified_viewer(args.eval_dir, args.port))