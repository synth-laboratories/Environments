from __future__ import annotations
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, Field
from dataclasses import asdict

from .engine import (
    TreasureHunterEngine,
    TreasureHunterPublicState,
    TreasureHunterPrivateState,
    TreasureHunterEngineSnapshot,
)
from .schema import TreasureHunterTaskInstance
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from src.reproducibility.core import ReproducibleEnvironment
from src.tasks.core import TaskInstance
from src.environment.tools import AbstractTool, EnvToolCall, ToolResult, TOOL_REGISTRY, register_tool


class CommandInput(BaseModel):
    command: str = Field(..., description="Text command like 'north' or 'take'")


class CommandTool(AbstractTool):
    name = "command"
    description = "Execute a text command in the treasure hunter world"
    call_schema = CommandInput
    result_schema = ToolResult

    def __init__(self, engine: TreasureHunterEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated = self.call_schema(**call.args)
            priv, pub = await self.engine._step_engine(validated.command)
            return ToolResult(ok=True, payload={"public": asdict(pub), "private": asdict(priv)})
        except Exception as e:
            pub = self.engine._build_public_state()
            return ToolResult(ok=False, error=str(e), payload={"public": asdict(pub)})


class TreasureObservationCallable(GetObservationCallable):
    async def get_observation(self, pub: TreasureHunterPublicState, priv: TreasureHunterPrivateState) -> InternalObservation:
        obs = asdict(pub)
        obs.update({
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        })
        return obs


class TreasureHunterEnvironment(StatefulEnvironment, ReproducibleEnvironment[TreasureHunterEngine]):
    def __init__(self, task_instance: TreasureHunterTaskInstance, custom_step_obs: Optional[GetObservationCallable] = None, custom_ckpt_obs: Optional[GetObservationCallable] = None):
        self.name = "TreasureHunter"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or TreasureObservationCallable()
        self.custom_checkpoint_observation_callable = custom_ckpt_obs or TreasureObservationCallable()
        self.engine = TreasureHunterEngine(task_instance)

        self._command_tool = CommandTool(self.engine)
        if self._command_tool.name not in TOOL_REGISTRY:
            register_tool(self._command_tool)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:
        self.engine.terminated = True
        priv, pub = self.engine._build_private_state(0.0), self.engine._build_public_state()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    def validate_tool_calls(self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]) -> EnvToolCall:
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Empty tool calls")
            first = tool_calls[0]
            if isinstance(first, list):
                first = first[0]
            call = first
        else:
            call = tool_calls
        if not isinstance(call, EnvToolCall):
            raise TypeError("tool call must be EnvToolCall")
        if call.tool != "command":
            raise ValueError("Unknown tool")
        return call

    async def step(self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]) -> InternalObservation:
        call = self.validate_tool_calls(tool_calls)
        result = await self._command_tool(call)
        if result.ok:
            priv = TreasureHunterPrivateState(**result.payload["private"])
            pub = TreasureHunterPublicState(**result.payload["public"])
        else:
            priv, pub = self.engine._build_private_state(0.0), self.engine._build_public_state()
            if pub.error_info is None:
                pub.error_info = result.error
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def checkpoint(self) -> InternalObservation:
        snapshot = await self.engine._serialize_engine()
        priv, pub = self.engine._build_private_state(0.0), self.engine._build_public_state()
        obs = await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)
        if isinstance(obs, dict):
            obs["engine_snapshot_data"] = snapshot.state
        return obs

    async def _to_observation(self, priv: TreasureHunterPrivateState, pub: TreasureHunterPublicState, obs_cb: Optional[GetObservationCallable]) -> InternalObservation:
        cb = obs_cb or TreasureObservationCallable()
        return await cb.get_observation(pub, priv)

    async def _serialize_engine(self) -> TreasureHunterEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, snapshot: TreasureHunterEngineSnapshot, task_instance: TaskInstance) -> "TreasureHunterEnvironment":
        eng = await TreasureHunterEngine._deserialize_engine(snapshot, task_instance)
        env = cls(task_instance)
        env.engine = eng
        return env
