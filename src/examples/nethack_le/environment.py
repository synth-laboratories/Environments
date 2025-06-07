from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
import nle.nethack as nethack

from examples.nethack_le.engine import (
    NethackEngine,
    NethackPrivateState,
    NethackPublicState,
    NethackEngineSnapshot,
)
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from src.reproducibility.core import ReproducibleEnvironment
from src.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
    TOOL_REGISTRY,
    register_tool,
)
from src.tasks.core import TaskInstance


# --- Observation Callables -------------------------------------------------
class NethackTextObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: NethackPublicState, priv: NethackPrivateState
    ) -> InternalObservation:
        screen = nethack.tty_render(pub.tty_chars, pub.tty_colors, pub.tty_cursor)
        return {
            "screen_text": screen,
            "message": pub.message,
            "reward_last": priv.reward_last_step,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
            "step": pub.step,
        }


class NethackImageTextObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: NethackPublicState, priv: NethackPrivateState
    ) -> InternalObservation:
        data = await NethackTextObservationCallable().get_observation(pub, priv)
        if pub.glyphs is not None:
            data["glyphs"] = pub.glyphs.tolist()
        return data


# --- Tools -----------------------------------------------------------------
class NethackActionInput(BaseModel):
    action: int = Field(..., description="NLE action index")


class NethackActionTool(AbstractTool):
    name = "nethack_action"
    description = "Execute an action in the Nethack environment"
    call_schema = NethackActionInput
    result_schema = ToolResult

    def __init__(self, engine: NethackEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            priv, pub = await self.engine._step_engine(validated_args.action)
            return ToolResult(
                ok=True,
                payload={"public": pub, "private": priv},
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


# --- Environment -----------------------------------------------------------
class NethackEnvironment(StatefulEnvironment, ReproducibleEnvironment[NethackEngine]):
    """Wrapper exposing NethackEngine via the standard environment API."""

    def __init__(
        self,
        task_instance: TaskInstance,
        mode: str = "text",
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "NethackLE"
        self.task_instance = task_instance
        self.mode = mode
        if mode == "image":
            default_cb = NethackImageTextObservationCallable()
        else:
            default_cb = NethackTextObservationCallable()
        self.custom_step_observation_callable = custom_step_obs or default_cb
        self.custom_checkpoint_observation_callable = custom_ckpt_obs or default_cb
        render_mode = "ansi" if mode in ("text", "image") else "human"
        self.engine = NethackEngine(task_instance, render_mode=render_mode)

        self.action_tool = NethackActionTool(self.engine)
        if self.action_tool.name not in TOOL_REGISTRY:
            register_tool(self.action_tool)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:
        screen = (
            self.engine.env.render() if self.engine.last_observation is not None else ""
        )
        return {"terminated": True, "final_screen": screen}

    def validate_tool_calls(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> EnvToolCall:
        raw_call_data: Dict[str, Any]
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Received empty list of tool calls.")
            first_item = tool_calls[0]
            if isinstance(first_item, list):
                if not first_item:
                    raise ValueError("Received empty inner list of tool calls.")
                raw_call_data = first_item[0]
            elif isinstance(first_item, dict):
                raw_call_data = first_item
            elif isinstance(first_item, EnvToolCall):
                return first_item
            else:
                raise TypeError(
                    f"Unexpected type in tool_calls list: {type(first_item)}"
                )
        elif isinstance(tool_calls, dict):
            raw_call_data = tool_calls
        elif isinstance(tool_calls, EnvToolCall):
            return tool_calls
        else:
            raise TypeError(f"Unexpected type for tool_calls: {type(tool_calls)}")

        if not isinstance(raw_call_data, dict):
            raise TypeError(f"Processed call data is not a dict: {type(raw_call_data)}")

        tool_name = raw_call_data.get("tool")
        tool_args = raw_call_data.get("args", {})
        if tool_name != "nethack_action":
            raise ValueError(f"Unknown tool: {tool_name}. Expected 'nethack_action'.")
        return EnvToolCall(tool=tool_name, args=tool_args)

    async def step(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> InternalObservation:
        agent_call = self.validate_tool_calls(tool_calls)
        tool_result: ToolResult = await self.action_tool(agent_call)

        if not tool_result.ok:
            raise RuntimeError(f"Tool failed: {tool_result.error}")
        priv_state = tool_result.payload["private"]
        pub_state = tool_result.payload["public"]
        return await self._to_observation(
            priv_state, pub_state, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:
        snap: NethackEngineSnapshot = await self.engine._serialize_engine()
        obs = {"engine_snapshot_data": snap}
        return obs

    async def _to_observation(
        self,
        priv: NethackPrivateState,
        pub: NethackPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (obs_cb or NethackTextObservationCallable()).get_observation(
            pub, priv
        )

    async def _serialize_engine(self) -> NethackEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: NethackEngineSnapshot, task_instance: TaskInstance
    ) -> "NethackEnvironment":
        eng = await NethackEngine._deserialize_engine(snapshot, task_instance)
        env = cls(task_instance)
        env.engine = eng
        return env
