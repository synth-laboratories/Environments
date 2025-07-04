#!/usr/bin/env python3
"""
Viewer V2 - Built from scratch with proper error handling and no assumptions.
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


class ViewerV2:
    """A viewer that actually works."""
    
    def __init__(self, eval_dir: Path):
        # ASSERTION: eval_dir must exist
        assert eval_dir.exists(), f"Evaluation directory does not exist: {eval_dir}"
        self.eval_dir = eval_dir
        
        # ASSERTION: traces directory must exist
        self.traces_dir = eval_dir / "traces"
        assert self.traces_dir.exists(), f"Traces directory does not exist: {self.traces_dir}"
        
        self.app = FastAPI()
        self.traces: Dict[str, Any] = {}
        self._setup_routes()
        
    def load_traces(self) -> Dict[str, str]:
        """Load all trace files with full error checking."""
        self.traces.clear()
        trace_files = list(self.traces_dir.glob("*.json"))
        
        print(f"🔍 Found {len(trace_files)} trace files in {self.traces_dir}")
        
        errors = []
        for trace_file in trace_files:
            try:
                print(f"  Loading {trace_file.name}...")
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                
                # ASSERTION: Must have expected structure
                assert isinstance(data, dict), f"Trace {trace_file} is not a dict"
                assert "trace" in data, f"Trace {trace_file} missing 'trace' key"
                assert isinstance(data["trace"], dict), f"Trace {trace_file} 'trace' is not a dict"
                
                trace_id = trace_file.stem
                self.traces[trace_id] = data
                print(f"    ✅ Loaded successfully")
                
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
        """Set up all routes with proper error handling."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main viewer page."""
            return """<!DOCTYPE html>
<html>
<head>
    <title>Synth Trace Viewer V2</title>
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .error { 
            color: #e74c3c; 
            background: #fee; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px;
            border-left: 4px solid #e74c3c;
        }
        .success { 
            color: #27ae60; 
            background: #efe; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px;
            border-left: 4px solid #27ae60;
        }
        .info { 
            color: #3498db; 
            background: #e3f2fd; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }
        button { 
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            margin: 5px;
            transition: background 0.2s;
        }
        button:hover { 
            background: #2980b9;
        }
        select { 
            background: white;
            color: #333;
            border: 1px solid #ddd;
            padding: 8px 12px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 14px;
        }
        pre { 
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            overflow: auto;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.4;
        }
        #status { 
            margin: 20px 0;
            padding: 15px;
            border-radius: 4px;
            font-weight: 500;
        }
        .section {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0;
            color: #2c3e50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Synth Trace Viewer V2</h1>
        <p class="subtitle">NO ASSUMPTIONS. ASSERTIONS EVERYWHERE.</p>
    
    <div id="status"></div>
    
    <div class="section">
        <h2>Step 1: Load Traces</h2>
        <button onclick="loadTraces()">LOAD TRACES</button>
        <div id="load-result"></div>
    </div>
    
    <div class="section">
        <h2>Step 2: Select Trace</h2>
        <select id="trace-select" onchange="selectTrace(this.value)">
            <option value="">-- No traces loaded --</option>
        </select>
        <div id="trace-info"></div>
    </div>
    
    <div class="section">
        <h2>Step 3: View Data</h2>
        <div id="trace-data"></div>
    </div>
    
    <script>
    let traces = {};
    let currentTrace = null;
    
    function showStatus(message, type = 'info') {
        const status = document.getElementById('status');
        status.className = type;
        status.textContent = message;
        console.log(`[${type}] ${message}`);
    }
    
    async function loadTraces() {
        showStatus('Loading traces...', 'info');
        
        try {
            const response = await fetch('/api/load-traces');
            const result = await response.json();
            
            console.log('Load result:', result);
            
            if (result.error) {
                showStatus(`Error: ${result.error}`, 'error');
                document.getElementById('load-result').innerHTML = `<pre>${result.details || ''}</pre>`;
                return;
            }
            
            // Get trace list
            const listResponse = await fetch('/api/traces');
            traces = await listResponse.json();
            
            console.log('Traces:', traces);
            
            // Update dropdown
            const select = document.getElementById('trace-select');
            select.innerHTML = '<option value="">-- Select a trace --</option>';
            
            for (const [traceId, info] of Object.entries(traces)) {
                const option = document.createElement('option');
                option.value = traceId;
                option.textContent = `${traceId} (${info.num_events} events)`;
                select.appendChild(option);
            }
            
            showStatus(`Loaded ${Object.keys(traces).length} traces successfully`, 'success');
            document.getElementById('load-result').innerHTML = `
                <div class="success">
                    ✅ Loaded: ${result.loaded}<br>
                    ❌ Errors: ${result.errors.length}
                    ${result.errors.length > 0 ? '<br>Errors:<pre>' + result.errors.join('\\n') + '</pre>' : ''}
                </div>
            `;
            
        } catch (error) {
            showStatus(`Failed to load traces: ${error.message}`, 'error');
            console.error('Load error:', error);
        }
    }
    
    async function selectTrace(traceId) {
        if (!traceId) {
            document.getElementById('trace-info').innerHTML = '';
            document.getElementById('trace-data').innerHTML = '';
            return;
        }
        
        showStatus(`Loading trace ${traceId}...`, 'info');
        
        try {
            const response = await fetch(`/api/trace/${traceId}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to load trace');
            }
            
            currentTrace = await response.json();
            console.log('Loaded trace:', currentTrace);
            
            // Display trace info
            const info = traces[traceId];
            document.getElementById('trace-info').innerHTML = `
                <div class="success">
                    <strong>Trace ID:</strong> ${traceId}<br>
                    <strong>Events:</strong> ${info.num_events}<br>
                    <strong>Has dataset:</strong> ${info.has_dataset ? 'Yes' : 'No'}<br>
                    <strong>Keys:</strong> ${info.keys.join(', ')}
                </div>
            `;
            
            // Display trace structure
            displayTraceData();
            
            showStatus(`Loaded trace ${traceId}`, 'success');
            
        } catch (error) {
            showStatus(`Failed to load trace: ${error.message}`, 'error');
            console.error('Trace load error:', error);
        }
    }
    
    function displayTraceData() {
        if (!currentTrace) return;
        
        let html = '<h3>Trace Structure</h3>';
        
        // Show top-level keys
        html += '<h4>Top Level Keys:</h4>';
        html += '<ul>';
        for (const key of Object.keys(currentTrace)) {
            html += `<li>${key} (${typeof currentTrace[key]})</li>`;
        }
        html += '</ul>';
        
        // Show trace metadata
        if (currentTrace.trace && currentTrace.trace.metadata) {
            html += '<h4>Trace Metadata:</h4>';
            html += '<pre>' + JSON.stringify(currentTrace.trace.metadata, null, 2) + '</pre>';
        }
        
        // Show partition info
        if (currentTrace.trace && currentTrace.trace.partition) {
            const partition = currentTrace.trace.partition;
            html += `<h4>Partition (${partition.length} items):</h4>`;
            
            // Show first partition item structure
            if (partition.length > 0) {
                html += '<p>First partition item structure:</p>';
                html += '<pre>' + JSON.stringify(partition[0], null, 2).substring(0, 1000) + '...</pre>';
            }
        }
        
        // Show dataset info
        if (currentTrace.dataset) {
            html += '<h4>Dataset:</h4>';
            html += '<pre>' + JSON.stringify(currentTrace.dataset, null, 2).substring(0, 500) + '...</pre>';
        }
        
        document.getElementById('trace-data').innerHTML = html;
    }
    
    // Auto-load traces on page load
    window.onload = () => {
        showStatus('Ready to load traces', 'info');
    };
    </script>
    </div>
</body>
</html>"""
        
        @self.app.get("/api/load-traces")
        async def load_traces_endpoint():
            """Load traces with full error reporting."""
            try:
                result = self.load_traces()
                return JSONResponse(result)
            except Exception as e:
                return JSONResponse({
                    "error": str(e),
                    "details": traceback.format_exc()
                }, status_code=500)
        
        @self.app.get("/api/traces")
        async def get_trace_list():
            """Get list of loaded traces."""
            trace_list = {}
            for trace_id, data in self.traces.items():
                try:
                    trace_info = {
                        "num_events": len(data.get("trace", {}).get("partition", [])),
                        "has_dataset": "dataset" in data,
                        "keys": list(data.keys())
                    }
                    trace_list[trace_id] = trace_info
                except Exception as e:
                    trace_list[trace_id] = {"error": str(e)}
            
            return JSONResponse(trace_list)
        
        @self.app.get("/api/trace/{trace_id}")
        async def get_trace(trace_id: str):
            """Get specific trace data."""
            if trace_id not in self.traces:
                raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
            
            return JSONResponse(self.traces[trace_id])
        
        @self.app.get("/favicon.ico")
        async def favicon():
            """Serve the Synth favicon."""
            favicon_path = Path(__file__).parent.parent.parent.parent / "favicon_synth.ico"
            if favicon_path.exists():
                with open(favicon_path, 'rb') as f:
                    return Response(content=f.read(), media_type="image/x-icon")
            else:
                raise HTTPException(status_code=404, detail="Favicon not found")


async def run_viewer_v2(eval_dir: Path, port: int = 9000):
    """Run the new viewer."""
    print("\n" + "="*60)
    print("🔥 VIEWER V2 - SCORCHED EARTH EDITION 🔥")
    print("="*60)
    print(f"Evaluation directory: {eval_dir}")
    print(f"Port: {port}")
    print("="*60 + "\n")
    
    try:
        viewer = ViewerV2(eval_dir)
        
        config = uvicorn.Config(
            viewer.app, 
            host="0.0.0.0", 
            port=port, 
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        print(f"\n🚀 Launching at http://localhost:{port}")
        print("   NO ASSUMPTIONS. ASSERTIONS EVERYWHERE.")
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
    
    parser = argparse.ArgumentParser(description="Viewer V2 - No assumptions")
    parser.add_argument("eval_dir", type=Path, help="Evaluation directory")
    parser.add_argument("--port", type=int, default=9000, help="Port to run on")
    
    args = parser.parse_args()
    
    asyncio.run(run_viewer_v2(args.eval_dir, args.port))