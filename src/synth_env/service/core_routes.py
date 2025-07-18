from fastapi import APIRouter, HTTPException, Body
from uuid import uuid4
from typing import Dict, Any, List, Optional
from types import SimpleNamespace
from pydantic import BaseModel
import os
import json
import pickle
import base64
import numpy as np
import tempfile
from dataclasses import dataclass

from synth_env.service.registry import get_environment_cls, list_supported_env_types
from synth_env.stateful.core import StatefulEnvironment
from synth_env.environment.tools import EnvToolCall

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


# Environment-specific task instance creation
@dataclass
class MinimalTaskInstanceMetadata:
    """Minimal metadata for environments that need it."""
    pass


@dataclass
class MinimalIntent:
    """Minimal intent for environments that need it."""
    rubric: Dict[str, Any]
    gold_trajectories: Optional[Any] = None
    gold_state_diff: Dict = None
    deterministic_eval_functions: list = None

    def __post_init__(self):
        if self.gold_state_diff is None:
            self.gold_state_diff = {}
        if self.deterministic_eval_functions is None:
            self.deterministic_eval_functions = []


@dataclass
class MinimalImpetus:
    """Minimal impetus for environments that need it."""
    instructions: str


def create_task_instance_for_environment(env_name: str, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create appropriate task instance for different environments."""
    
    if env_name in ["Sokoban", "CrafterClassic", "MiniGrid", "TicTacToe"]:
        # These environments work with SimpleNamespace
        task = SimpleNamespace(initial_engine_snapshot=initial_state or {})
        
        # For MiniGrid, handle seed-based environment selection
        if env_name == "MiniGrid" and config:
            # Check if a seed is provided in config
            if "seed" in config:
                task.initial_engine_snapshot["seed"] = config["seed"]
            
            # Check if a specific environment is requested
            if "env_name" in config:
                task.initial_engine_snapshot["env_name"] = config["env_name"]
        
        return task
    
    elif env_name == "Verilog":
        # Verilog needs a snapshot_dir attribute
        # Create a temporary directory for the snapshot
        temp_dir = tempfile.mkdtemp(prefix="verilog_task_")
        task = SimpleNamespace(
            initial_engine_snapshot=initial_state,
            snapshot_dir=temp_dir,
            metadata=MinimalTaskInstanceMetadata(),
            id=uuid4()
        )
        return task
    
    elif env_name == "NetHack":
        # NetHack needs proper TaskInstance structure with NetHackTaskInstanceMetadata
        from synth_env.examples.nethack.taskset import NetHackTaskInstanceMetadata
        
        metadata = NetHackTaskInstanceMetadata(
            character_role="tourist",  # Easy starting character
            starting_level=1,
            target_depth=3,
            time_limit=1000,
            difficulty="tutorial",
            special_objectives=["Explore at least 3 different dungeon levels"],
            seed=42
        )
        
        task = SimpleNamespace(
            initial_engine_snapshot=initial_state,
            metadata=metadata,
            id=uuid4(),
            intent=MinimalIntent(rubric={"success": "reach target depth"}),
            impetus=MinimalImpetus(instructions="Play NetHack and achieve the highest score."),
            is_reproducible=False
        )
        return task
    
    elif env_name == "Enron":
        # Enron needs task instance with email data
        # For now, provide minimal structure
        task = SimpleNamespace(
            initial_engine_snapshot=initial_state,
            metadata=MinimalTaskInstanceMetadata(),
            id=uuid4(),
            # Enron might need specific data structure
            question=initial_state.get("question", "What information can you find?") if initial_state else "What information can you find?",
            answer=initial_state.get("answer", "") if initial_state else "",
            emails=initial_state.get("emails", []) if initial_state else []
        )
        return task
    
    else:
        # Default: use SimpleNamespace for unknown environments
        return SimpleNamespace(initial_engine_snapshot=initial_state)


async def reconstruct_task_instance_from_serialized(env_name: str, serialized_data: Dict[str, Any]) -> Any:
    """Reconstruct a task instance from serialized data for specific environment types."""
    
    if env_name == "MiniGrid":
        # MiniGrid has its own TaskInstance class with deserialize method
        from synth_env.examples.minigrid.taskset import MiniGridTaskInstance
        return await MiniGridTaskInstance.deserialize(serialized_data)
    
    elif env_name == "Sokoban":
        # Sokoban has its own TaskInstance class with deserialize method
        from synth_env.examples.sokoban.taskset import SokobanTaskInstance
        return await SokobanTaskInstance.deserialize(serialized_data)
    
    elif env_name in ["CrafterClassic", "TicTacToe"]:
        # These environments work with SimpleNamespace - convert serialized data back to SimpleNamespace
        from types import SimpleNamespace
        from uuid import UUID
        
        task = SimpleNamespace()
        task.id = UUID(serialized_data.get("id", str(uuid4())))
        task.initial_engine_snapshot = serialized_data.get("initial_engine_snapshot", {})
        task.metadata = SimpleNamespace(**serialized_data.get("metadata", {}))
        
        # Handle impetus
        impetus_data = serialized_data.get("impetus", {})
        if impetus_data:
            task.impetus = SimpleNamespace(instructions=impetus_data.get("instructions", ""))
        
        # Handle intent
        intent_data = serialized_data.get("intent", {})
        if intent_data:
            task.intent = SimpleNamespace(
                rubric=intent_data.get("rubric", ""),
                gold_trajectories=intent_data.get("gold_trajectories", []),
                gold_state_diff=intent_data.get("gold_state_diff", {})
            )
        
        task.is_reproducible = serialized_data.get("is_reproducible", True)
        
        return task
    
    elif env_name == "Verilog":
        # Verilog needs special handling with snapshot_dir
        from types import SimpleNamespace
        from uuid import UUID
        import tempfile
        
        task = SimpleNamespace()
        task.id = UUID(serialized_data.get("id", str(uuid4())))
        task.initial_engine_snapshot = serialized_data.get("initial_engine_snapshot", {})
        task.metadata = MinimalTaskInstanceMetadata()
        task.snapshot_dir = tempfile.mkdtemp(prefix="verilog_task_")
        
        # Handle impetus
        impetus_data = serialized_data.get("impetus", {})
        if impetus_data:
            task.impetus = SimpleNamespace(instructions=impetus_data.get("instructions", ""))
        
        # Handle intent
        intent_data = serialized_data.get("intent", {})
        if intent_data:
            task.intent = SimpleNamespace(
                rubric=intent_data.get("rubric", ""),
                gold_trajectories=intent_data.get("gold_trajectories", []),
                gold_state_diff=intent_data.get("gold_state_diff", {})
            )
        
        task.is_reproducible = serialized_data.get("is_reproducible", True)
        
        return task
    
    elif env_name == "NetHack":
        # NetHack needs proper TaskInstance structure with NetHackTaskInstanceMetadata
        from synth_env.examples.nethack.taskset import NetHackTaskInstanceMetadata
        from types import SimpleNamespace
        from uuid import UUID
        
        # Extract metadata from serialized data
        metadata_data = serialized_data.get("metadata", {})
        metadata = NetHackTaskInstanceMetadata(
            character_role=metadata_data.get("character_role", "tourist"),
            starting_level=metadata_data.get("starting_level", 1),
            target_depth=metadata_data.get("target_depth", 3),
            time_limit=metadata_data.get("time_limit", 1000),
            difficulty=metadata_data.get("difficulty", "tutorial"),
            special_objectives=metadata_data.get("special_objectives", ["Explore at least 3 different dungeon levels"]),
            seed=metadata_data.get("seed", 42)
        )
        
        task = SimpleNamespace()
        task.id = UUID(serialized_data.get("id", str(uuid4())))
        task.initial_engine_snapshot = serialized_data.get("initial_engine_snapshot", {})
        task.metadata = metadata
        
        # Handle impetus
        impetus_data = serialized_data.get("impetus", {})
        if impetus_data:
            task.impetus = MinimalImpetus(instructions=impetus_data.get("instructions", "Play NetHack and achieve the highest score."))
        else:
            task.impetus = MinimalImpetus(instructions="Play NetHack and achieve the highest score.")
        
        # Handle intent
        intent_data = serialized_data.get("intent", {})
        if intent_data:
            task.intent = MinimalIntent(
                rubric=intent_data.get("rubric", {"success": "reach target depth"}),
                gold_trajectories=intent_data.get("gold_trajectories", []),
                gold_state_diff=intent_data.get("gold_state_diff", {})
            )
        else:
            task.intent = MinimalIntent(rubric={"success": "reach target depth"})
        
        task.is_reproducible = serialized_data.get("is_reproducible", False)
        
        return task
    
    elif env_name == "Enron":
        # Enron needs task instance with email data
        from types import SimpleNamespace
        from uuid import UUID
        
        task = SimpleNamespace()
        task.id = UUID(serialized_data.get("id", str(uuid4())))
        task.initial_engine_snapshot = serialized_data.get("initial_engine_snapshot", {})
        task.metadata = MinimalTaskInstanceMetadata()
        
        # Enron-specific fields
        task.question = serialized_data.get("question", "What information can you find?")
        task.answer = serialized_data.get("answer", "")
        task.emails = serialized_data.get("emails", [])
        
        # Handle impetus
        impetus_data = serialized_data.get("impetus", {})
        if impetus_data:
            task.impetus = SimpleNamespace(instructions=impetus_data.get("instructions", ""))
        
        # Handle intent
        intent_data = serialized_data.get("intent", {})
        if intent_data:
            task.intent = SimpleNamespace(
                rubric=intent_data.get("rubric", ""),
                gold_trajectories=intent_data.get("gold_trajectories", []),
                gold_state_diff=intent_data.get("gold_state_diff", {})
            )
        
        task.is_reproducible = serialized_data.get("is_reproducible", True)
        
        return task
    
    else:
        # Default: use SimpleNamespace for unknown environments
        from types import SimpleNamespace
        from uuid import UUID
        
        task = SimpleNamespace()
        task.id = UUID(serialized_data.get("id", str(uuid4())))
        task.initial_engine_snapshot = serialized_data.get("initial_engine_snapshot", {})
        
        # Handle impetus
        impetus_data = serialized_data.get("impetus", {})
        if impetus_data:
            task.impetus = SimpleNamespace(instructions=impetus_data.get("instructions", ""))
        
        # Handle intent
        intent_data = serialized_data.get("intent", {})
        if intent_data:
            task.intent = SimpleNamespace(
                rubric=intent_data.get("rubric", ""),
                gold_trajectories=intent_data.get("gold_trajectories", []),
                gold_state_diff=intent_data.get("gold_state_diff", {})
            )
        
        task.is_reproducible = serialized_data.get("is_reproducible", True)
        
        return task


# Storage abstraction
class InstanceStorage:
    """Abstract storage for environment instances"""

    async def store(self, env_id: str, env: StatefulEnvironment):
        """Store an environment instance"""
        # ALWAYS store in-memory as fallback
        instances[env_id] = env
        
        # ALSO try to store in Redis if available (but don't rely on it)
        if REDIS_AVAILABLE and redis_client:
            try:
                # Serialize the environment using pickle and base64 encode
                serialized = base64.b64encode(pickle.dumps(env)).decode("utf-8")
                await redis_client.set(
                    f"env_instance:{env_id}", serialized, ex=3600
                )  # 1 hour TTL
                print(f"✅ Stored environment {env_id} in Redis + in-memory")
            except Exception as e:
                print(f"⚠️ Redis storage failed, using in-memory fallback: {e}")
        else:
            print(f"✅ Stored environment {env_id} in-memory (Redis not available)")

    async def get(self, env_id: str) -> Optional[StatefulEnvironment]:
        """Retrieve an environment instance"""
        # Try in-memory first (most reliable)
        if env_id in instances:
            print(f"✅ Retrieved environment {env_id} from in-memory store")
            return instances[env_id]
        
        # Fallback to Redis if not in memory
        if REDIS_AVAILABLE and redis_client:
            try:
                serialized = await redis_client.get(f"env_instance:{env_id}")
                if serialized:
                    # Deserialize from base64 and pickle
                    env = pickle.loads(base64.b64decode(serialized))
                    print(f"✅ Retrieved environment {env_id} from Redis (restored to memory)")
                    # Store back in memory for next time
                    instances[env_id] = env
                    return env
            except Exception as e:
                print(f"⚠️ Redis retrieval failed: {e}")
        
        print(f"❌ Environment {env_id} not found in either store")
        return None

    async def remove(self, env_id: str) -> Optional[StatefulEnvironment]:
        """Remove and return an environment instance"""
        # Get the environment first
        env = await self.get(env_id)
        
        # Remove from in-memory store
        removed_env = instances.pop(env_id, None)
        
        # Also try to remove from Redis
        if REDIS_AVAILABLE and redis_client:
            try:
                await redis_client.delete(f"env_instance:{env_id}")
                print(f"✅ Removed environment {env_id} from both Redis and in-memory")
            except Exception as e:
                print(f"⚠️ Redis removal failed, removed from in-memory: {e}")
        else:
            print(f"✅ Removed environment {env_id} from in-memory")
        
        return env or removed_env


# Global storage instance
storage = InstanceStorage()


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    from dataclasses import is_dataclass
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif is_dataclass(obj):
        # Handle dataclasses safely - check if they have a to_dict method first
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            # Fallback to converting __dict__ but exclude numpy arrays to prevent recursion
            result = {}
            for key, value in obj.__dict__.items():
                if not isinstance(value, np.ndarray):
                    result[key] = convert_numpy_types(value)
                else:
                    result[key] = value.tolist()  # Convert numpy arrays directly
            return result
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        # Handle other objects with __dict__ but be more cautious
        try:
            # Only process if it's likely to be a simple object
            if len(obj.__dict__) < 50:  # Avoid overly complex objects
                result = {}
                for key, value in obj.__dict__.items():
                    if not isinstance(value, np.ndarray):
                        result[key] = convert_numpy_types(value)
                    else:
                        result[key] = value.tolist()
                return result
            else:
                return str(obj)  # Fallback to string representation
        except (RecursionError, AttributeError):
            return str(obj)  # Safe fallback
    else:
        return obj


# Request/Response models for better API documentation
class InitializeRequest(BaseModel):
    initial_state: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    task_instance: Optional[Dict[str, Any]] = None  # Add task_instance field


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
    import traceback
    
    try:
        print(f"🔍 Initializing {env_name} environment...")
        
        cls = get_environment_cls(env_name)
        print(f"✅ Got environment class: {cls}")

        # Handle task_instance parameter - use it if provided, otherwise create a new one
        if request.task_instance:
            print(f"🔍 Using provided task_instance...")
            task = await reconstruct_task_instance_from_serialized(env_name, request.task_instance)
            print(f"✅ Reconstructed task instance: {type(task)}")
        else:
            print(f"🔍 Creating new task instance...")
            # Create environment-specific task instance
            task = create_task_instance_for_environment(env_name, request.initial_state, request.config)
            print(f"✅ Created task instance: {type(task)}")
        
        # This is where recursion might happen for Sokoban
        print(f"🔍 Creating environment instance...")
        env = cls(task)
        print(f"✅ Created environment instance")

        # Generate unique environment ID
        env_id = str(uuid4())
        print(f"✅ Generated env_id: {env_id}")

        # Initialize and get first observation - this might also cause recursion
        print(f"🔍 Calling env.initialize()...")
        obs = await env.initialize()
        print(f"✅ Environment initialized, observation type: {type(obs)}")

        # Store the fully initialized environment (fixes Redis initialization bug)
        print(f"🔍 Storing environment...")
        await storage.store(env_id, env)
        print(f"✅ Environment stored")

        # Convert numpy types to Python types for JSON serialization
        print(f"🔍 Converting numpy types...")
        obs_serializable = convert_numpy_types(obs)
        print(f"✅ Numpy types converted")

        return {"env_id": env_id, "observation": obs_serializable, "done": False, "info": {}}
    
    except RecursionError as e:
        # Capture recursion errors specifically
        stack_trace = traceback.format_exc()
        print(f"❌ RECURSION ERROR in {env_name} initialization:")
        print(stack_trace)
        raise HTTPException(status_code=400, detail=f"Recursion error during {env_name} initialization: {str(e)}")
    
    except Exception as e:
        # Capture all other errors
        stack_trace = traceback.format_exc()
        print(f"❌ ERROR in {env_name} initialization:")
        print(stack_trace)
        raise HTTPException(status_code=400, detail=f"Error during {env_name} initialization: {str(e)}")


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
        raw_tool_calls = request.action.get("tool_calls", [])
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Extracted raw_tool_calls = {raw_tool_calls}",
            file=sys.stderr,
        )
        
        # Convert dictionaries to EnvToolCall objects
        tool_calls = []
        for call_dict in raw_tool_calls:
            if isinstance(call_dict, dict):
                # Convert dict to EnvToolCall object
                tool_call = EnvToolCall(
                    tool=call_dict.get("tool", ""),
                    args=call_dict.get("args", {})
                )
                tool_calls.append(tool_call)
            else:
                # Already an EnvToolCall object
                tool_calls.append(call_dict)
        
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Converted to EnvToolCall objects: {tool_calls}",
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
        # FIX: StatefulEnvironment.step() returns observation dict directly,
        # not a dict with 'observation', 'reward', 'done', 'info' keys
        response = {
            "observation": result,  # result IS the observation
            "reward": result.get("reward_last", None),  # Try to get reward from obs
            "done": result.get("terminated", False) or result.get("truncated", False),
            "info": {"terminated": result.get("terminated", False), "truncated": result.get("truncated", False)},
        }
        
        # Convert numpy types to Python types for JSON serialization
        response_serializable = convert_numpy_types(response)
        
        print(
            f"🌐 ENVIRONMENTS SERVICE {request_id}: Returning response with keys: {list(response_serializable.keys())}",
            file=sys.stderr,
        )
        return response_serializable
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
        observation_serializable = convert_numpy_types(observation)

        return {
            "public": observation_serializable,
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
    task = create_task_instance_for_environment(env_type, initial_state, config)
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
    obs_serializable = convert_numpy_types(obs)
    return {"private": obs_serializable, "public": obs_serializable}


@api_router.post("/{env_type}/{instance_id}/step", deprecated=True)
async def step_env_legacy(
    env_type: str, instance_id: str, calls: List[Any]
) -> Dict[str, Any]:
    """[DEPRECATED] Use /env/{env_name}/step instead."""
    env = await storage.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.step(calls)
    obs_serializable = convert_numpy_types(obs)
    return {"private": obs_serializable, "public": obs_serializable}


@api_router.post("/{env_type}/{instance_id}/terminate", deprecated=True)
async def terminate_env_legacy(env_type: str, instance_id: str) -> Any:
    """[DEPRECATED] Use /env/{env_name}/terminate instead."""
    env = await storage.remove(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    obs = await env.terminate()
    obs_serializable = convert_numpy_types(obs)
    return obs_serializable


@api_router.get("/{env_type}/{instance_id}/checkpoint")
async def checkpoint_env(env_type: str, instance_id: str) -> Dict[str, Any]:
    """Get a checkpoint of the environment state."""
    env = await storage.get(instance_id)
    if not env:
        raise HTTPException(status_code=404, detail="Instance not found")
    snapshot = await env.checkpoint()
    snapshot_serializable = convert_numpy_types(snapshot)
    return {"snapshot": snapshot_serializable}
