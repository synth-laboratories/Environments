from fastapi import APIRouter, HTTPException, Body
from uuid import uuid4
from typing import Dict, Any, List, Optional
from types import SimpleNamespace
from pydantic import BaseModel
import os
import json
import pickle
import base64

from synth_env.service.registry import get_environment_cls, list_supported_env_types
from synth_env.stateful.core import StatefulEnvironment

# Try to import Redis for persistent storage
try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
    # Create Redis client
    redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        encoding="utf-8",
        decode_responses=False,  # We need binary mode for pickle
    )
except ImportError:
    REDIS_AVAILABLE = False
    redis_client = None

api_router = APIRouter()

# Fallback in-memory store if Redis is not available
instances: Dict[str, StatefulEnvironment] = {}


# Storage abstraction
class InstanceStorage:
    """Abstract storage for environment instances"""

    async def store(self, env_id: str, env: StatefulEnvironment):
        """Store an environment instance"""
        if REDIS_AVAILABLE and redis_client:
            try:
                # Serialize the environment using pickle and base64 encode
                serialized = base64.b64encode(pickle.dumps(env)).decode("utf-8")
                await redis_client.set(
                    f"env_instance:{env_id}", serialized, ex=3600
                )  # 1 hour TTL
                print(f"✅ Stored environment {env_id} in Redis")
            except Exception as e:
                print(f"⚠️ Redis storage failed, using in-memory: {e}")
                instances[env_id] = env
        else:
            instances[env_id] = env

    async def get(self, env_id: str) -> Optional[StatefulEnvironment]:
        """Retrieve an environment instance"""
        if REDIS_AVAILABLE and redis_client:
            try:
                serialized = await redis_client.get(f"env_instance:{env_id}")
                if serialized:
                    # Deserialize from base64 and pickle
                    env = pickle.loads(base64.b64decode(serialized))
                    print(f"✅ Retrieved environment {env_id} from Redis")
                    return env
                else:
                    print(
                        f"❌ Environment {env_id} not found in Redis, checking in-memory store"
                    )
                    return instances.get(env_id)
            except Exception as e:
                print(f"⚠️ Redis retrieval failed, checking in-memory: {e}")
                return instances.get(env_id)
        else:
            return instances.get(env_id)

    async def remove(self, env_id: str) -> Optional[StatefulEnvironment]:
        """Remove and return an environment instance"""
        if REDIS_AVAILABLE and redis_client:
            try:
                # Get the environment first
                env = await self.get(env_id)
                if env:
                    await redis_client.delete(f"env_instance:{env_id}")
                    print(f"✅ Removed environment {env_id} from Redis")
                return env
            except Exception as e:
                print(f"⚠️ Redis removal failed, using in-memory: {e}")
                return instances.pop(env_id, None)
        else:
            return instances.pop(env_id, None)


# Global storage instance
storage = InstanceStorage()


# Request/Response models for better API documentation
class InitializeRequest(BaseModel):
    initial_state: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    env_id: str
    request_id: str
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

        # Initialize and get first observation
        obs = await env.initialize()

        # Store the fully initialized environment (fixes Redis initialization bug)
        await storage.store(env_id, env)

        return {"env_id": env_id, "observation": obs, "done": False, "info": {}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/env/{env_name}/step")
async def step_env(env_name: str, request: StepRequest = Body(...)) -> Dict[str, Any]:
    """Execute a step in the environment."""
    import uuid as uuid_module
    import sys

    # Generate unique request ID for this HTTP request
    request_id = str(uuid_module.uuid4())[:8]
    print(
        f"🌐 ENVIRONMENTS SERVICE {request_id}: request_id = {request_id}",
        file=sys.stderr,
    )
    print(
        f"\n🌐 ENVIRONMENTS SERVICE {request_id}: step_env HTTP endpoint called",
        file=sys.stderr,
    )
    print(
        f"🌐 ENVIRONMENTS SERVICE {request_id}: env_name = {env_name}", file=sys.stderr
    )
    print(
        f"🌐 ENVIRONMENTS SERVICE {request_id}: env_id = {request.env_id}",
        file=sys.stderr,
    )
    print(
        f"🌐 ENVIRONMENTS SERVICE {request_id}: action = {request.action}",
        file=sys.stderr,
    )

    # Log call stack to see where this HTTP request comes from
    import traceback

    stack = traceback.format_stack()
    print(
        f"🌐 ENVIRONMENTS SERVICE {request_id}: Call stack (last 3 frames):",
        file=sys.stderr,
    )
    for frame in stack[-3:]:
        print(f"  {frame.strip()}", file=sys.stderr)

    print(
        f"🌐 ENVIRONMENTS SERVICE {request_id}: About to retrieve environment from storage",
        file=sys.stderr,
    )
    env = await storage.get(request.env_id)
    if not env:
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Environment not found!",
            file=sys.stderr,
        )
        raise HTTPException(
            status_code=404, detail=f"Environment instance {request.env_id} not found"
        )

    try:
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: About to extract tool calls from action",
            file=sys.stderr,
        )
        # Extract tool calls from action
        tool_calls = request.action.get("tool_calls", [])
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Extracted tool_calls = {tool_calls}",
            file=sys.stderr,
        )

        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: About to call env.step()",
            file=sys.stderr,
        )
        # Execute step
        result = await env.step(tool_calls)
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: env.step() completed, result type = {type(result)}",
            file=sys.stderr,
        )

        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: About to store environment back to storage",
            file=sys.stderr,
        )
        # Store the updated environment state
        await storage.store(request.env_id, env)
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Environment stored successfully",
            file=sys.stderr,
        )

        # Format response
        response = {
            "observation": result.get("observation", {}),
            "reward": result.get("reward", None),
            "done": result.get("done", False),
            "info": result.get("info", {}),
        }
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Returning response with keys: {list(response.keys())}",
            file=sys.stderr,
        )
        return response
    except Exception as e:
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Exception during step: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/env/{env_name}/terminate")
async def terminate_env(
    env_name: str, request: TerminateRequest = Body(...)
) -> Dict[str, Any]:
    """Terminate an environment instance."""
    env = await storage.remove(request.env_id)
    if not env:
        raise HTTPException(
            status_code=404, detail=f"Environment instance {request.env_id} not found"
        )

    try:
        # Terminate environment and capture observation
        observation = await env.terminate()

        return {
            "public": observation,
            "private": {"instance_id": request.env_id},
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

    # Initialize the environment before storing (fixes Redis initialization bug)
    await env.initialize()
    await storage.store(instance_id, env)
    return {"instance_id": instance_id}


@api_router.post("/{env_type}/{instance_id}/reset", deprecated=True)
async def reset_env_legacy(
    env_type: str, instance_id: str, seed: Optional[int] = None
) -> Dict[str, Any]:
    """[DEPRECATED] Use /env/{env_name}/initialize instead."""
    env = await storage.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.initialize()
    return {"private": obs, "public": obs}


@api_router.post("/{env_type}/{instance_id}/step", deprecated=True)
async def step_env_legacy(
    env_type: str, instance_id: str, calls: List[Any]
) -> Dict[str, Any]:
    """[DEPRECATED] Use /env/{env_name}/step instead."""
    env = await storage.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.step(calls)
    return {"private": obs, "public": obs}


@api_router.post("/{env_type}/{instance_id}/terminate", deprecated=True)
async def terminate_env_legacy(env_type: str, instance_id: str) -> Any:
    """[DEPRECATED] Use /env/{env_name}/terminate instead."""
    env = await storage.remove(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.terminate()
    return obs


@api_router.get("/{env_type}/{instance_id}/checkpoint")
async def checkpoint_env(env_type: str, instance_id: str) -> Dict[str, Any]:
    """Get a checkpoint of the environment state."""
    env = await storage.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    snapshot = await env.checkpoint()
    return {"snapshot": snapshot}
