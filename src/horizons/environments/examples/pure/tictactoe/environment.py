import dataclasses
import sys
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from horizons.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from horizons.environments.environment.tools import (
    TOOL_REGISTRY,
    AbstractTool,
    EnvToolCall,
    ToolResult,
    register_tool,
)
from horizons.environments.reproducibility.core import ReproducibleEnvironment
from horizons.environments.stateful.core import StatefulEnvironment
from horizons.environments.tasks.core import TaskInstance

from .engine import (
    SynthTicTacToeCheckpointObservationCallable,
    SynthTicTacToeObservationCallable,
    TicTacToeEngine,
    TicTacToeEngineSnapshot,
    TicTacToePrivateState,
    TicTacToePublicState,
)


# --- Tool Definition ---
class TicTacToeActionInput(BaseModel):
    letter: str  # "A", "B", or "C"
    number: int  # 1, 2, or 3


class TicTacToeInteractTool(AbstractTool):
    name = "interact"
    description = "Place your mark (X or O) in the specified cell using letter (A, B, C) and number (1, 2, 3) coordinates."
    call_schema = TicTacToeActionInput
    result_schema = ToolResult

    def __init__(self, engine: TicTacToeEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Parse input - now using separate letter and number parameters
            letter = call.args.get("letter")
            number = call.args.get("number")

            if not letter or number is None:
                return ToolResult(
                    ok=False,
                    error="Both 'letter' and 'number' parameters are required.",
                    payload={},
                )

            # Validate letter
            if letter.upper() not in ["A", "B", "C"]:
                return ToolResult(
                    ok=False,
                    error=f"Invalid letter '{letter}'. Must be A, B, or C.",
                    payload={},
                )

            # Validate number
            if number not in [1, 2, 3]:
                return ToolResult(
                    ok=False,
                    error=f"Invalid number '{number}'. Must be 1, 2, or 3.",
                    payload={},
                )

            # Convert to coordinate format (e.g., "A1", "B2", etc.)
            coord = f"{letter.upper()}{number}"

            # Execute the move
            priv_state, pub_state = await self.engine._step_engine(coord)

            return ToolResult(
                ok=True,
                payload={
                    "public": pub_state.to_dict(),
                    "private": priv_state.to_dict(),
                },
            )
        except Exception as e:
            # Add current public state to payload for context in case of error
            _, pub_state_on_error = self.engine.get_current_states_for_observation()
            return ToolResult(
                ok=False,
                error=str(e),
                payload={"public": pub_state_on_error.to_dict()},
            )


class TicTacToeEnvironment(StatefulEnvironment, ReproducibleEnvironment[TicTacToeEngine]):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "TicTacToe"
        self.task_instance = task_instance
        # Default to SynthTicTacToeObservationCallable if none provided
        self.custom_step_observation_callable = custom_step_obs or SynthTicTacToeObservationCallable()
        self.custom_checkpoint_observation_callable = (
            custom_ckpt_obs or SynthTicTacToeCheckpointObservationCallable()
        )
        self.engine: TicTacToeEngine = TicTacToeEngine(task_instance)

        self._interact_tool = TicTacToeInteractTool(self.engine)
        if self._interact_tool.name not in TOOL_REGISTRY:
            register_tool(self._interact_tool)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        priv.terminated = True  # Mark as terminated
        obs_dict = {"terminated": True, "message": "Game terminated."}
        # Use _to_observation to format, including final state
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable, extra_obs=obs_dict
        )

    async def step(self, tool_calls: List[EnvToolCall]) -> InternalObservation:
        # Validate tool calls first
        self.validate_tool_calls(tool_calls)

        results = []
        for tool_call in tool_calls:
            if tool_call.tool == "interact":
                result = await self._interact_tool(tool_call)
                results.append(result)
            else:
                # Handle other tools if implemented
                pass

        # After all tool calls, get the current observation
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    def validate_tool_calls(self, tool_calls: List[EnvToolCall]) -> None:
        for tool_call in tool_calls:
            if tool_call.tool not in ["interact"]:
                raise ValueError(f"Unknown tool: {tool_call.tool}")

            # Validate required parameters
            args = tool_call.args or {}
            letter = args.get("letter")
            number = args.get("number")

            if not letter or number is None:
                raise ValueError("Both 'letter' and 'number' parameters are required for interact tool")

            if str(letter).upper() not in ["A", "B", "C"]:
                raise ValueError(f"Invalid letter '{letter}'. Must be A, B, or C")

            if number not in [1, 2, 3]:
                raise ValueError(f"Invalid number '{number}'. Must be 1, 2, or 3")

    async def checkpoint(self) -> InternalObservation:
        priv, pub = self.engine.get_current_states_for_observation()
        return await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )

    async def _to_observation(
        self,
        priv: TicTacToePrivateState,
        pub: TicTacToePublicState,
        obs_callable: GetObservationCallable,
        extra_obs: Optional[Dict[str, Any]] = None,
    ) -> InternalObservation:
        # Get the observation from the callable
        obs = await obs_callable.get_observation(pub, priv)

        # Add extra observation data if provided
        if extra_obs:
            if hasattr(obs, 'public_observation') and isinstance(obs.public_observation, dict):
                obs.public_observation.update(extra_obs)

        return obs
