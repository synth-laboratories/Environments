"""Neural MMO Classic Environment wrapper.

Bridges tool-calls from agents to NeuralMMOEngine dynamics, supporting both
image and vector observation modes. Converts the engine's state dataclasses
into InternalObservation format for the evaluation pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
import dataclasses

from synth_env.examples.nmmo_classic.engine import (
    NeuralMMOEngine,
    NeuralMMOObservationCallable,
    NeuralMMOPublicState,
    NeuralMMOPrivateState,
)
from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.stateful.core import StatefulEnvironment
from synth_env.reproducibility.core import ReproducibleEnvironment
from synth_env.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
    register_tool,
)
from synth_env.tasks.core import TaskInstance


# Tool input schemas
class NeuralMMOActionInput(BaseModel):
    action: Dict[str, Any] = Field(
        ..., description="NMMO action dictionary with keys like 'Move', 'Attack', etc."
    )


# Tool definitions
class NeuralMMOActionTool(AbstractTool):
    name = "nmmo_action"
    description = "Execute an action in the Neural MMO environment"
    call_schema = NeuralMMOActionInput
    result_schema = ToolResult

    def __init__(self, engine: NeuralMMOEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            priv_state, pub_state = await self.engine._step_engine(
                validated_args.action
            )
            return ToolResult(
                ok=True,
                payload={
                    "public": dataclasses.asdict(pub_state),
                    "private": dataclasses.asdict(priv_state),
                },
            )
        except Exception as e:
            # Get current state for error context
            if self.engine._last_public_state and self.engine._last_private_state:
                pub_state = self.engine._last_public_state
                priv_state = self.engine._last_private_state
            else:
                # Create minimal error state
                pub_state = self.engine._build_public_state({})
                priv_state = self.engine._build_private_state(0.0, {}, False, {})
            return ToolResult(
                ok=False,
                error=str(e),
                payload={"public": dataclasses.asdict(pub_state)},
            )


class NeuralMMOEnvironment(
    StatefulEnvironment, ReproducibleEnvironment[NeuralMMOEngine]
):
    """
    Neural MMO Environment with dual observation mode support.

    Responsibilities:
        • Hold a live NeuralMMOEngine instance
        • Convert tool-calls → validated NMMO actions
        • Support both image and vector observation modes
        • Expose .initialize / .step / .checkpoint / .terminate
        • Forward (de)serialization to the underlying engine
    """

    # ------------------------------------------------------------------ #
    # ctor / lifecycle helpers                                           #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        task_instance: TaskInstance,
        observation_mode: str = "vector",
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "NeuralMMO"
        self.task_instance = task_instance
        self.observation_mode = observation_mode
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        self.engine: NeuralMMOEngine = NeuralMMOEngine(task_instance, observation_mode)

        # Register tools for this environment
        self.action_tool = NeuralMMOActionTool(self.engine)
        register_tool(self.action_tool)

    async def initialize(self) -> InternalObservation:  # type: ignore[override]
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:  # type: ignore[override]
        # Nothing to flush – just acknowledge termination
        return {"terminated": True, "message": "Environment terminated."}  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # step / checkpoint                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_tool_calls(tool_calls: List[List[EnvToolCall]]) -> None:
        if not tool_calls or not isinstance(tool_calls[0][0], EnvToolCall):
            raise ValueError("tool_calls must be a nested list of EnvToolCall objects")

    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:  # type: ignore[override]
        self._validate_tool_calls(tool_calls)

        # Execute the tool call - should be nmmo_action
        tool_call = tool_calls[0][0]

        if tool_call.name == "nmmo_action":
            result = await self.action_tool(tool_call)
            if result.ok:
                # Extract states from tool result
                pub_dict = result.payload["public"]
                priv_dict = result.payload["private"]

                # Convert back to dataclass instances
                pub = NeuralMMOPublicState(**pub_dict)
                priv = NeuralMMOPrivateState(**priv_dict)

                return await self._to_observation(
                    priv, pub, self.custom_step_observation_callable
                )
            else:
                # Handle tool execution error
                raise ValueError(f"Tool execution failed: {result.error}")
        else:
            # Fallback to direct action execution for backward compatibility
            if hasattr(tool_call, "action"):
                action_payload: Dict[str, Any] = tool_call.action  # type: ignore[attr-defined]
                priv, pub = await self.engine._step_engine(action_payload)
                return await self._to_observation(
                    priv, pub, self.custom_step_observation_callable
                )
            else:
                raise ValueError(f"Unknown tool: {tool_call.name}")

    async def checkpoint(self) -> InternalObservation:  # type: ignore[override]
        """
        A lightweight snapshot mid-episode – use the engine's stored public/private state.
        """
        # Retrieve last stored public and private states
        priv = self.engine._last_private_state
        pub = self.engine._last_public_state
        # Delegate to the same observation pipeline as step/initialize
        return await self._to_observation(
            priv,
            pub,
            self.custom_checkpoint_observation_callable,
        )

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    async def _to_observation(
        self,
        priv: NeuralMMOPrivateState,
        pub: NeuralMMOPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (
            obs_cb or NeuralMMOObservationCallable(self.engine)
        ).get_observation(pub, priv)

    # ------------------------------------------------------------------ #
    # ReproducibleEnvironment plumbing                                   #
    # ------------------------------------------------------------------ #
    async def _serialize_engine(self) -> Any:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, snapshot: Any) -> "NeuralMMOEnvironment":
        eng = await NeuralMMOEngine._deserialize_engine(snapshot)
        env = cls(eng.task_instance, eng._observation_mode)
        env.engine = eng
        return env
