from typing import Optional, List

from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from src.environment.tools import EnvToolCall

from .tools import SubmitAnswerTool
from .engine import (
    FinanceEngine,
    FinancePrivateState,
    FinancePublicState,
    FinanceEngineSnapshot,
)
from .schema import FinanceTaskInstance


class FinanceEnv(StatefulEnvironment):
    def __init__(
        self,
        task_instance: FinanceTaskInstance,
        custom_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "ValsFinance"
        self.task_instance = task_instance
        self.custom_obs = custom_obs
        self.engine = FinanceEngine(task_instance)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_obs(priv, pub)

    async def terminate(self) -> InternalObservation:
        return {"terminated": True, "message": "Environment session terminated."}

    def validate_tool_calls(self, tool_calls: List[List[EnvToolCall]]) -> None:
        if (
            not tool_calls
            or not tool_calls[0]
            or not isinstance(tool_calls[0][0], SubmitAnswerTool)
        ):
            raise ValueError(
                "Expected SubmitAnswerTool wrapped in a non-empty nested list."
            )

    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:
        self.validate_tool_calls(tool_calls)
        answer = tool_calls[0][0].answer  # type: ignore[attr-defined]
        priv, pub = await self.engine._step_engine(answer)
        return await self._to_obs(priv, pub)

    async def checkpoint(self) -> InternalObservation:
        snap = await self.engine._serialize_engine()
        return {"engine_snapshot": snap.task_instance_dict, "message": "Checkpoint"}

    async def _to_obs(
        self, priv: FinancePrivateState, pub: FinancePublicState
    ) -> InternalObservation:
        obs = {
            "question": pub.question,
            "submitted_answer": priv.submitted_answer,
            "is_correct": priv.is_correct,
            "terminated": priv.terminated,
        }
        return obs
