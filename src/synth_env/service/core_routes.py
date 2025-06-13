from fastapi import APIRouter, HTTPException, Body
from uuid import uuid4
from typing import Dict, Any, List, Optional
from types import SimpleNamespace
from pydantic import BaseModel

from synth_env.service.registry import get_environment_cls, list_supported_env_types
from synth_env.stateful.core import StatefulEnvironment

api_router = APIRouter()

# In-memory store of live environment instances
instances: Dict[str, StatefulEnvironment] = {}


# Request/Response models for better API documentation
class InitializeRequest(BaseModel):
    initial_state: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    env_id: str
    action: Dict[str, Any]


class TerminateRequest(BaseModel):
    env_id: str


@api_router.get("/health")
async def get_health():
    return {"status": "ok", "supported_environments": list_supported_env_types()}


@api_router.post("/env/{env_name}/initialize")
async def initialize_env(
    env_name: str, request: InitializeRequest = Body(...)
) -> Dict[str, Any]:
    """Initialize a new environment instance."""
    try:
        cls = get_environment_cls(env_name)

        # Create a minimal task object carrying initial_engine_snapshot for snapshot-based environments
        task = SimpleNamespace(initial_engine_snapshot=request.initial_state)
        env = cls(task)

        # Generate unique environment ID
        env_id = str(uuid4())
        instances[env_id] = env

        # Initialize and get first observation
        obs = await env.initialize()

        return {"env_id": env_id, "observation": obs, "done": False, "info": {}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/env/{env_name}/step")
async def step_env(env_name: str, request: StepRequest = Body(...)) -> Dict[str, Any]:
    """Execute a step in the environment."""
    env = instances.get(request.env_id)
    if not env:
        raise HTTPException(
            status_code=404, detail=f"Environment instance {request.env_id} not found"
        )

    try:
        # Extract tool calls from action
        tool_calls = request.action.get("tool_calls", [])

        # Execute step
        result = await env.step(tool_calls)

        # Format response
        return {
            "observation": result.get("observation", {}),
            "reward": result.get("reward", None),
            "done": result.get("done", False),
            "info": result.get("info", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/env/{env_name}/terminate")
async def terminate_env(
    env_name: str, request: TerminateRequest = Body(...)
) -> Dict[str, Any]:
    """Terminate an environment instance."""
    env = instances.pop(request.env_id, None)
    if not env:
        raise HTTPException(
            status_code=404, detail=f"Environment instance {request.env_id} not found"
        )

    try:
        # Terminate environment
        await env.terminate()

        return {
            "success": True,
            "message": f"Environment {request.env_id} terminated successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Keep backward compatibility endpoints but mark as deprecated
@api_router.post("/{env_type}/create", deprecated=True)
async def create_env_legacy(
    env_type: str,
    config: Optional[Dict[str, Any]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """[DEPRECATED] Use /env/{env_name}/initialize instead."""
    cls = get_environment_cls(env_type)
    task = SimpleNamespace(initial_engine_snapshot=initial_state)
    env = cls(task)
    instance_id = str(uuid4())
    instances[instance_id] = env
    return {"instance_id": instance_id}


@api_router.post("/{env_type}/{instance_id}/reset", deprecated=True)
async def reset_env_legacy(
    env_type: str, instance_id: str, seed: Optional[int] = None
) -> Dict[str, Any]:
    """[DEPRECATED] Use /env/{env_name}/initialize instead."""
    env = instances.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.initialize()
    return {"private": obs, "public": obs}


@api_router.post("/{env_type}/{instance_id}/step", deprecated=True)
async def step_env_legacy(
    env_type: str, instance_id: str, calls: List[Any]
) -> Dict[str, Any]:
    """[DEPRECATED] Use /env/{env_name}/step instead."""
    env = instances.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.step(calls)
    return {"private": obs, "public": obs}


@api_router.post("/{env_type}/{instance_id}/terminate", deprecated=True)
async def terminate_env_legacy(env_type: str, instance_id: str) -> Any:
    """[DEPRECATED] Use /env/{env_name}/terminate instead."""
    env = instances.pop(instance_id, None)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.terminate()
    return obs


@api_router.get("/{env_type}/{instance_id}/checkpoint")
async def checkpoint_env(env_type: str, instance_id: str) -> Dict[str, Any]:
    """Get a checkpoint of the environment state."""
    env = instances.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    snapshot = await env.checkpoint()
    return {"snapshot": snapshot}
