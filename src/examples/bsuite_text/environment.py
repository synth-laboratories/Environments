from __future__ import annotations

import dataclasses
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from src.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
    TOOL_REGISTRY,
    register_tool,
)
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment

from .engine import (
    BSuiteEngine,
    BSuitePublicState,
    BSuitePrivateState,
    BSuiteEngineSnapshot,
)
from .taskset import BSuiteTaskInstance


class BSuiteActionInput(BaseModel):
    action: int = Field(..., description="Integer action for the bsuite environment")


class BSuiteActionTool(AbstractTool):
    name = "bsuite_action"
    description = "Execute an action in the bsuite environment"
    call_schema = BSuiteActionInput
    result_schema = ToolResult

    def __init__(self, engine: BSuiteEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            priv, pub = await self.engine._step_engine(validated_args.action)
            return ToolResult(
                ok=True,
                payload={
                    "public": dataclasses.asdict(pub),
                    "private": dataclasses.asdict(priv),
                },
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class BSuiteObservationCallable(GetObservationCallable):
    async def get_observation(self, pub: BSuitePublicState, priv: BSuitePrivateState) -> InternalObservation:
        obs = dataclasses.asdict(pub)
        obs["total_reward"] = priv.total_reward
        obs["terminated"] = priv.terminated
        return obs


class BSuiteTextEnvironment(StatefulEnvironment):
    """Stateful environment exposing DeepMind bsuite via text-only tools."""

    def __init__(
        self,
        task_instance: BSuiteTaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "BSuiteText"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or BSuiteObservationCallable()
        self.custom_checkpoint_observation_callable = custom_ckpt_obs or BSuiteObservationCallable()
        self.engine = BSuiteEngine(task_instance)

        self._action_tool = BSuiteActionTool(self.engine)
        if self._action_tool.name not in TOOL_REGISTRY:
            register_tool(self._action_tool)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:
        priv = BSuitePrivateState(total_reward=self.engine.total_reward, terminated=True)
        pub = BSuitePublicState(observation=None, step_type=-1, reward=0.0, discount=0.0)
        obs = await self._to_observation(priv, pub, self.custom_step_observation_callable)
        if isinstance(obs, dict):
            obs["terminated"] = True
        return obs

    def validate_tool_calls(
        self,
        tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]], dict, int, List[int]],
    ) -> EnvToolCall:
        action: Optional[int] = None
        if isinstance(tool_calls, EnvToolCall):
            if tool_calls.tool == self._action_tool.name:
                return tool_calls
            action = tool_calls.args.get("action")
        elif isinstance(tool_calls, list):
            first = tool_calls[0]
            if isinstance(first, list):
                first = first[0]
            if isinstance(first, EnvToolCall):
                if first.tool == self._action_tool.name:
                    return first
                action = first.args.get("action")
            elif isinstance(first, dict):
                action = first.get("action")
            else:
                action = int(first)
        elif isinstance(tool_calls, dict):
            action = tool_calls.get("action")
        elif isinstance(tool_calls, int):
            action = tool_calls

        if action is None:
            raise ValueError("Missing 'action' for bsuite step call")

        return EnvToolCall(tool=self._action_tool.name, args={"action": int(action)})

    async def step(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]], dict, int, List[int]]
    ) -> InternalObservation:
        call = self.validate_tool_calls(tool_calls)
        result = await self._action_tool(call)
        if not result.ok or not isinstance(result.payload, dict):
            raise ValueError(result.error or "Tool execution failed")
        priv = BSuitePrivateState(**result.payload["private"])
        pub = BSuitePublicState(**result.payload["public"])
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def checkpoint(self) -> InternalObservation:
        snapshot: BSuiteEngineSnapshot = await self.engine._serialize_engine()
        priv = BSuitePrivateState(total_reward=self.engine.total_reward, terminated=False)
        pub = BSuitePublicState(observation=None, step_type=-1, reward=0.0, discount=0.0)
        obs = await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)
        if isinstance(obs, dict):
            obs["engine_snapshot_data"] = dataclasses.asdict(snapshot)
        return obs

    async def _to_observation(
        self,
        priv: BSuitePrivateState,
        pub: BSuitePublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (obs_cb or BSuiteObservationCallable()).get_observation(pub, priv)
