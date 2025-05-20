from fastapi import APIRouter, HTTPException
from uuid import uuid4
from typing import Dict, Any, List
from types import SimpleNamespace

from service.registry import get_environment_cls, list_supported_env_types
from tasks.core import TaskInstance
from stateful.core import StatefulEnvironment

api_router = APIRouter()

# In-memory store of live environment instances
instances: Dict[str, StatefulEnvironment] = {}

@api_router.get("/health")
async def get_health():
    return {"status": "ok", "supported_environments": list_supported_env_types()}

@api_router.post("/{env_type}/create")
async def create_env(env_type: str, config: Dict[str, Any] = None, initial_state: Dict[str, Any] = None):
    cls = get_environment_cls(env_type)
    # Create a minimal task object carrying initial_engine_snapshot for snapshot-based environments
    task = SimpleNamespace(initial_engine_snapshot=initial_state)
    env = cls(task)
    instance_id = str(uuid4())
    instances[instance_id] = env
    return {"instance_id": instance_id}

@api_router.post("/{env_type}/{instance_id}/reset")
async def reset_env(env_type: str, instance_id: str, seed: int = None):
    env = instances.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    # Environment returns a single observation dict; wrap it for old private/public API
    obs = await env.initialize()
    return {"private": obs, "public": obs}

@api_router.post("/{env_type}/{instance_id}/step")
async def step_env(env_type: str, instance_id: str, calls: List[Any]):
    env = instances.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    # Environment returns a single observation dict; wrap it for old private/public API
    obs = await env.step(calls)
    return {"private": obs, "public": obs}

@api_router.post("/{env_type}/{instance_id}/terminate")
async def terminate_env(env_type: str, instance_id: str):
    env = instances.pop(instance_id, None)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.terminate()
    return obs

@api_router.get("/{env_type}/{instance_id}/checkpoint")
async def checkpoint_env(env_type: str, instance_id: str):
    env = instances.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    snapshot = await env.checkpoint()
    return {"snapshot": snapshot}
