from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from horizons.environments.environment.shared_engine import GetObservationCallable, InternalObservation
from horizons.environments.environment.tools import EnvToolCall, ToolResult
from horizons.environments.stateful.core import StatefulEnvironment


class SokobanActionInput(BaseModel):
    action: int


class SokobanPyO3Environment(StatefulEnvironment):
    def __init__(
        self,
        task_instance: SimpleNamespace,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "Sokoban_PyO3"
        self.task_instance = task_instance
        try:
            import horizons_env_py as hep
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Failed to import horizons_env_py (build the PyO3 module): {e}"
            )
        # Create registry-backed Rust env with optional config (seed, dims, boxes, max_steps)
        cfg: Optional[Dict[str, Any]] = None
        if hasattr(task_instance, "config") and isinstance(task_instance.config, dict):
            cfg = dict(task_instance.config)
        elif hasattr(task_instance, "initial_engine_snapshot") and isinstance(task_instance.initial_engine_snapshot, dict):
            snap = task_instance.initial_engine_snapshot
            # Map common keys if present
            cfg = {
                k: v
                for k, v in {
                    "seed": snap.get("seed"),
                    "width": snap.get("width") or (snap.get("dim_room", [None, None])[0] if isinstance(snap.get("dim_room"), list) else None),
                    "height": snap.get("height") or (snap.get("dim_room", [None, None])[1] if isinstance(snap.get("dim_room"), list) else None),
                    "num_boxes": snap.get("num_boxes"),
                    "max_steps": snap.get("max_steps"),
                    "use_simple_level": snap.get("use_simple_level"),
                    "room_fixed": snap.get("room_fixed"),
                    "room_state": snap.get("room_state"),
                }.items()
                if v is not None
            }
            if not cfg:
                cfg = None
        self._hep = hep
        self._env = hep.create_rust_env("Sokoban", cfg)

    async def initialize(self) -> InternalObservation:
        return self._env.initialize()

    async def terminate(self) -> InternalObservation:
        return self._env.terminate()

    def validate_tool_calls(
        self,
        tool_calls: Union[EnvToolCall, List[Dict[str, Any]], Dict[str, Any]],
    ) -> EnvToolCall:
        # Accept a single EnvToolCall directly
        if isinstance(tool_calls, EnvToolCall):
            return tool_calls
        # Accept a list with either dicts or EnvToolCall instances
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("empty tool_calls")
            first = tool_calls[0]
            if isinstance(first, EnvToolCall):
                return first
            if not isinstance(first, dict):
                raise TypeError(f"tool_call must be dict-like, got {type(first)}")
            obj = first
        elif isinstance(tool_calls, dict):
            obj = tool_calls
        else:
            raise TypeError("invalid tool_calls type")
        tool = obj.get("tool")
        if tool != "interact":
            raise ValueError(f"Unknown tool: {tool}")
        args = obj.get("args", {})
        return EnvToolCall(tool=tool, args=args)

    async def step(self, tool_calls: Union[EnvToolCall, List[Dict[str, Any]], Dict[str, Any]]) -> InternalObservation:
        call = self.validate_tool_calls(tool_calls)
        # Forward directly to PyO3 generic step
        return self._env.step_tool("interact", call.args)

    async def checkpoint(self) -> InternalObservation:
        # Map checkpoint to a regular observation shape for compatibility
        snap = self._env.checkpoint()
        # Return a minimal observation embedding the snapshot
        return {
            "public_observation": {"event": "checkpoint", "engine_snapshot": snap},
            "private_observation": {},
        }
