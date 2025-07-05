from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import json
from pathlib import Path

from .database import (
    list_evaluations,
    list_traces,
    get_trace,
    get_trace_from_file,
    get_environments,
)
from .models import Evaluation, TraceMeta, Trace, TraceEvent
from .live import live

app = FastAPI(title="Synth Trace Viewer API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Synth Trace Viewer API", "version": "2.0.0"}


@app.get("/environments", response_model=List[str])
async def environments():
    """Get list of all available environments."""
    try:
        return get_environments()
    except Exception as e:
        # Fallback to filesystem
        evals_dir = Path("../evals")
        if evals_dir.exists():
            return [d.name for d in evals_dir.iterdir() if d.is_dir()]
        return []


@app.get("/evaluations", response_model=List[Evaluation])
async def evaluations(
    env: Optional[str] = Query(None, description="Filter by environment"),
):
    """Get list of evaluations, optionally filtered by environment."""
    try:
        df = list_evaluations(env)
        return df.to_dict("records")
    except Exception as e:
        # Fallback to filesystem
        evaluations = []
        base_dir = Path("../evals")

        if env:
            env_dirs = [base_dir / env]
        else:
            env_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        for env_dir in env_dirs:
            if not env_dir.exists():
                continue

            for run_dir in env_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    # Try to load evaluation summary
                    summary_file = run_dir / "evaluation_summary.json"
                    if summary_file.exists():
                        with open(summary_file, "r") as f:
                            summary = json.load(f)
                            evaluations.append(
                                Evaluation(
                                    env_name=env_dir.name,
                                    run_id=run_dir.name,
                                    eval_timestamp=summary.get("timestamp", ""),
                                    models_evaluated=",".join(
                                        summary.get("models", [])
                                    ),
                                    num_trajectories=summary.get("num_trajectories", 0),
                                    success_rate=summary.get("success_rate", 0.0),
                                    metadata=json.dumps(summary.get("metadata", {})),
                                )
                            )

        return evaluations


@app.get("/traces/{env_name}/{run_id}", response_model=List[TraceMeta])
async def traces(env_name: str, run_id: str):
    """Get list of traces for a specific evaluation run."""
    try:
        df = list_traces(run_id)
        return df.to_dict("records")
    except Exception as e:
        # Fallback to filesystem
        traces_dir = Path(f"../evals/{env_name}/{run_id}/traces")
        if not traces_dir.exists():
            raise HTTPException(
                status_code=404, detail=f"Traces not found for {env_name}/{run_id}"
            )

        traces = []
        for trace_file in traces_dir.glob("*.json"):
            # Load just enough to get metadata
            with open(trace_file, "r") as f:
                trace_data = json.load(f)

            trace_meta = trace_data.get("trace", {}).get("metadata", {})
            dataset = trace_data.get("dataset", {})

            traces.append(
                TraceMeta(
                    trace_id=trace_file.stem,
                    trajectory_id=trace_file.stem,
                    model_name=trace_meta.get("model_name", "unknown"),
                    total_reward=dataset.get("reward_signals", [{}])[0].get(
                        "reward", 0.0
                    ),
                    num_steps=len(trace_data.get("trace", {}).get("partition", [])),
                    terminated_reason=trace_meta.get("termination_reason", None),
                )
            )

        return traces


@app.get("/trace/{env_name}/{run_id}/{trace_id}", response_model=Trace)
async def trace(env_name: str, run_id: str, trace_id: str):
    """Get full trace data."""
    try:
        # Try database first
        trace_data = get_trace(trace_id)
        if trace_data:
            return Trace(
                trace_id=trace_id,
                trajectory_id=trace_data.get("trajectory_id", trace_id),
                model_name=trace_data.get("model_name", "unknown"),
                total_reward=trace_data.get("total_reward", 0.0),
                num_steps=trace_data.get("num_steps", 0),
                trace=trace_data.get("trace", {}),
            )
    except Exception:
        pass

    # Fallback to filesystem
    trace_data = get_trace_from_file(env_name, run_id, trace_id)
    if not trace_data:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

    # Extract metadata from trace
    trace_meta = trace_data.get("trace", {}).get("metadata", {})
    dataset = trace_data.get("dataset", {})

    return Trace(
        trace_id=trace_id,
        trajectory_id=trace_id,
        model_name=trace_meta.get("model_name", "unknown"),
        total_reward=dataset.get("reward_signals", [{}])[0].get("reward", 0.0),
        num_steps=len(trace_data.get("trace", {}).get("partition", [])),
        trace=trace_data,
        env_name=env_name,
        run_id=run_id,
    )


@app.get("/trace/image/{env_name}/{run_id}/{trace_id}/{turn}/{step}")
async def trace_image(env_name: str, run_id: str, trace_id: str, turn: int, step: int):
    """Get image for a specific turn/step."""
    # This would serve images from file system or extract from trace
    # For now, return 404
    raise HTTPException(status_code=404, detail="Image serving not yet implemented")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for live trace updates."""
    await live.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Could handle client commands here
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        live.disconnect(websocket)


@app.post("/notify/new_trace")
async def notify_new_trace(event: TraceEvent):
    """Notify clients about new trace (called by evaluation system)."""
    await live.broadcast_new_trace(
        trace_id=event.trace_id,
        run_id=event.run_id or "unknown",
        env_name=event.env_name or "unknown",
    )
    return {"status": "broadcast sent"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
