from typing import Optional, List

from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from src.environment.tools import EnvToolCall

from examples.scicode.engine import (
    SciCodeEngine,
    SciCodePublicState,
    SciCodePrivateState,
    SciCodeEngineSnapshot,
    SynthSciCodeObservationCallable,
)
from examples.scicode.schema import SciCodeTaskInstance
from examples.scicode.tools import SubmitAnswerTool


class SciCodeEnv(StatefulEnvironment):
    def __init__(
        self,
        task_instance: SciCodeTaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "SciCodeEnv"
        self.task_instance = task_instance
        self.custom_obs = custom_step_obs or SynthSciCodeObservationCallable()
        self.engine = SciCodeEngine(task_instance)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self.custom_obs.get_observation(pub, priv)

    async def terminate(self) -> InternalObservation:
        return {"terminated": True, "message": "Environment terminated"}

    def validate_tool_calls(self, tool_calls: List[List[EnvToolCall]]) -> None:
        if (
            not tool_calls
            or not tool_calls[0]
            or not isinstance(tool_calls[0][0], SubmitAnswerTool)
        ):
            raise ValueError("Expected SubmitAnswerTool")

    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:
        self.validate_tool_calls(tool_calls)
        answer = tool_calls[0][0].answer  # type: ignore[attr-defined]
        priv, pub = await self.engine._step_engine(answer)
        return await self.custom_obs.get_observation(pub, priv)

    async def checkpoint(self) -> InternalObservation:
        snap = await self.engine._serialize_engine()
        return snap.model_dump()

    @classmethod
    async def _deserialize_engine(cls, snapshot: SciCodeEngineSnapshot) -> "SciCodeEnv":
        engine = await SciCodeEngine._deserialize_engine(snapshot)
        env = cls(task_instance=engine.task_instance)
        env.engine = engine
        return env
