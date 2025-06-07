from typing import List, Optional

from examples.memory_text.engine import (
    MemoryTextEngine,
    MemoryTextPrivateState,
    MemoryTextPublicState,
    MemoryTextEngineSnapshot,
)
from examples.memory_text.tools import RecallSequenceTool
from examples.memory_text.schema import MemoryTextTaskInstance

from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.reproducibility.core import ReproducibleEnvironment
from src.stateful.core import StatefulEnvironment
from src.environment.tools import EnvToolCall


class TextMemoryGymEnv(StatefulEnvironment, ReproducibleEnvironment[MemoryTextEngine]):
    def __init__(
        self,
        task_instance: MemoryTextTaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "TextMemoryGym"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        self.engine = MemoryTextEngine(task_instance.metadata.sequence_length)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:
        return {"terminated": True}

    def validate_tool_calls(self, tool_calls: List[EnvToolCall]) -> RecallSequenceTool:
        if not tool_calls:
            raise ValueError("No tool call provided")
        call = tool_calls[0]
        if not isinstance(call, RecallSequenceTool):
            raise ValueError("Expected RecallSequenceTool")
        return call

    async def step(self, tool_calls: List[EnvToolCall]) -> InternalObservation:
        call = self.validate_tool_calls(tool_calls)
        priv, pub = await self.engine._step_engine(call.answer)
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:
        snapshot = await self.engine._serialize_engine()
        priv, pub = self.engine.get_current_states_for_observation()
        obs = await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )
        if isinstance(obs, dict):
            obs["engine_snapshot_data"] = snapshot.model_dump()
        return obs

    async def _to_observation(
        self,
        priv: MemoryTextPrivateState,
        pub: MemoryTextPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        if obs_cb:
            return await obs_cb.get_observation(pub, priv)
        return {
            "prompt": pub.prompt,
            "terminated": pub.terminated,
            "correct": pub.correct,
        }

    async def _serialize_engine(self) -> MemoryTextEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: MemoryTextEngineSnapshot, task_instance: MemoryTextTaskInstance
    ) -> "TextMemoryGymEnv":
        engine = await MemoryTextEngine._deserialize_engine(snapshot)
        env = cls(task_instance)
        env.engine = engine
        return env
