from typing import Optional, Any, Dict, Union, List
from pydantic import BaseModel

from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from src.environment.tools import AbstractTool, EnvToolCall, ToolResult, TOOL_REGISTRY, register_tool
from src.tasks.core import TaskInstance

from .engine import (
    MiniHackEngine,
    MiniHackTextObservationCallable,
    MiniHackVLMObservationCallable,
    MiniHackPrivateState,
    MiniHackPublicState,
)
from .taskset import MiniHackTaskInstance


class MiniHackActionInput(BaseModel):
    action: int


class MiniHackInteractTool(AbstractTool):
    name = "interact"
    description = "Perform an action in the MiniHack environment"
    call_schema = MiniHackActionInput
    result_schema = ToolResult

    def __init__(self, engine: MiniHackEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            priv, pub = await self.engine._step_engine(validated_args.action)
            return ToolResult(ok=True, payload={"public": pub, "private": priv})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class MiniHackEnvironment(StatefulEnvironment):
    def __init__(
        self,
        task_instance: Optional[MiniHackTaskInstance] = None,
        vlm_mode: bool = False,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "MiniHack"
        from .taskset import INSTANCE as DEFAULT_TASK
        self.task_instance = task_instance or DEFAULT_TASK
        self.engine = MiniHackEngine(self.task_instance)
        self.vlm_mode = vlm_mode
        self.custom_step_observation_callable = (
            custom_step_obs or (
                MiniHackVLMObservationCallable() if vlm_mode else MiniHackTextObservationCallable()
            )
        )
        self.custom_checkpoint_observation_callable = custom_ckpt_obs or self.custom_step_observation_callable
        self._interact_tool = MiniHackInteractTool(self.engine)
        if self._interact_tool.name not in TOOL_REGISTRY:
            register_tool(self._interact_tool)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def terminate(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        priv.terminated = True
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    def validate_tool_calls(
        self,
        tool_calls: Union[EnvToolCall, List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, Any]],
    ) -> EnvToolCall:
        raw_call: Dict[str, Any]
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Empty tool call list")
            first = tool_calls[0]
            if isinstance(first, list):
                if not first:
                    raise ValueError("Empty inner tool call list")
                raw_call = first[0]
            elif isinstance(first, dict):
                raw_call = first
            elif isinstance(first, EnvToolCall):
                return first
            else:
                raise TypeError("Unexpected tool call type")
        elif isinstance(tool_calls, dict):
            raw_call = tool_calls
        elif isinstance(tool_calls, EnvToolCall):
            return tool_calls
        else:
            raise TypeError("Unexpected tool call type")
        if raw_call.get("tool") != "interact":
            raise ValueError("Unknown tool")
        return EnvToolCall(tool="interact", args=raw_call.get("args", {}))

    async def step(
        self, tool_calls: Union[EnvToolCall, List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, Any]]
    ) -> InternalObservation:
        agent_call = self.validate_tool_calls(tool_calls)
        result = await self._interact_tool(agent_call)
        if not result.ok or not isinstance(result.payload, dict):
            priv, pub = await self.engine._reset_engine()
        else:
            priv = result.payload["private"]
            pub = result.payload["public"]
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)

    async def checkpoint(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_checkpoint_observation_callable)

    async def _to_observation(
        self,
        priv: MiniHackPrivateState,
        pub: MiniHackPublicState,
        obs_cb: GetObservationCallable,
    ) -> InternalObservation:
        return await obs_cb.get_observation(pub, priv)
