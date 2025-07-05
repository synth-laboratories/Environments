from __future__ import annotations

from typing import Dict, Optional, Any, List, Union
from pydantic import BaseModel

from synth_env.stateful.core import StatefulEnvironment
from synth_env.reproducibility.core import ReproducibleEnvironment
from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
)
from synth_env.tasks.core import TaskInstance

from .engine import (
    TicTacToeEngine,
    TicTacToePublicState,
    TicTacToePrivateState,
    TicTacToeEngineSnapshot,
    SynthTicTacToeObservationCallable,
    SynthTicTacToeCheckpointObservationCallable,
)


class TicTacToeActionInput(BaseModel):
    action: str  # "A1", "B2", etc.


class TicTacToeInteractTool(AbstractTool):
    name = "interact"
    description = "Place your mark (X or O) in the specified cell."
    call_schema = TicTacToeActionInput
    result_schema = ToolResult

    def __init__(self, engine: TicTacToeEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Parse input
            action = call.args.get("action")

            if not action:
                return ToolResult(ok=False, error="No action provided", payload={})

            # Execute action
            private_state, public_state = await self.engine._step_engine(action)

            return ToolResult(
                ok=True,
                payload={"public_state": public_state, "private_state": private_state},
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e), payload={})


class TicTacToeEnvironment(
    StatefulEnvironment, ReproducibleEnvironment[TicTacToeEngine]
):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "TicTacToe"
        self.task_instance = task_instance
        self.custom_step_observation_callable = (
            custom_step_obs or SynthTicTacToeObservationCallable()
        )
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthTicTacToeCheckpointObservationCallable()
        )
        self.engine = TicTacToeEngine(task_instance)
        self._interact_tool = TicTacToeInteractTool(self.engine)

    async def initialize(self) -> InternalObservation:
        # Reset engine and return initial observation
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def step(self, tool_calls) -> InternalObservation:
        # Validate and normalize tool calls
        validated_call = self.validate_tool_calls(tool_calls)

        # Execute the interact tool
        result = await self._interact_tool(validated_call)

        if result.ok:
            priv = result.payload["private_state"]
            pub = result.payload["public_state"]
            return await self._to_observation(
                priv, pub, self.custom_step_observation_callable
            )
        else:
            # Return error observation
            priv, pub = self.engine.get_current_states_for_observation()
            return await self._to_observation(
                priv,
                pub,
                self.custom_step_observation_callable,
                extra_obs={"error": result.error},
            )

    async def checkpoint(self) -> InternalObservation:
        # Return checkpoint observation
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )

    async def terminate(self) -> InternalObservation:
        # Mark as terminated and return final observation
        priv, pub = self.engine.get_current_states_for_observation()
        pub.terminated = True
        priv.terminated = True
        return await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )

    def validate_tool_calls(self, tool_calls) -> EnvToolCall:
        # Handle various input formats
        if isinstance(tool_calls, EnvToolCall):
            validated_call = tool_calls
        elif isinstance(tool_calls, dict):
            # Handle dict format
            if "tool" in tool_calls:
                validated_call = EnvToolCall(
                    tool=tool_calls["tool"], args=tool_calls.get("args", {})
                )
            elif "name" in tool_calls:
                # Handle legacy format
                validated_call = EnvToolCall(
                    tool=tool_calls["name"], args=tool_calls.get("parameters", {})
                )
            elif "function" in tool_calls:
                # Handle OpenAI function call format
                validated_call = EnvToolCall(
                    tool=tool_calls["function"]["name"],
                    args=tool_calls["function"].get("arguments", {}),
                )
            else:
                # Assume it's just parameters
                validated_call = EnvToolCall(tool="interact", args=tool_calls)
        elif isinstance(tool_calls, list):
            # Take first call from list
            if len(tool_calls) > 0:
                validated_call = self.validate_tool_calls(tool_calls[0])
            else:
                raise ValueError("Empty tool calls list")
        else:
            # Try to convert to dict
            validated_call = EnvToolCall(
                tool="interact", args={"action": str(tool_calls)}
            )

        # Validate tool name
        if validated_call.tool != "interact":
            raise ValueError(f"Unknown tool: {validated_call.tool}")

        return validated_call

    async def _to_observation(
        self,
        priv: TicTacToePrivateState,
        pub: TicTacToePublicState,
        obs_cb: Optional[GetObservationCallable],
        extra_obs: Optional[Dict] = None,
    ) -> InternalObservation:
        # Convert states to observation using callback
        if obs_cb:
            obs = await obs_cb.get_observation(pub, priv)
        else:
            obs: InternalObservation = {}

        if extra_obs and isinstance(obs, dict):
            obs.update(extra_obs)

        return obs

    async def _serialize_engine(self) -> TicTacToeEngineSnapshot:
        # Delegate to engine serialization
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: TicTacToeEngineSnapshot, task_instance: TaskInstance
    ) -> "TicTacToeEnvironment":
        # Create new environment instance
        env = cls(task_instance)
        # Restore engine from snapshot
        env.engine = await TicTacToeEngine._deserialize_engine(snapshot)
        # Update tool reference
        env._interact_tool = TicTacToeInteractTool(env.engine)
        return env
