from typing import List, Optional, Any, Dict
from examples.sokoban.engine import (
    SokobanEngine,
    SynthSokobanObservationCallable,
    SokobanPrivateState,
    SokobanPublicState,
)
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.reproducibility.core import ReproducibleEnvironment
from src.stateful.core import StatefulEnvironment
from src.tasks.core import TaskInstance
from src.environment.tools import EnvToolCall


class SokobanEnvironment(StatefulEnvironment, ReproducibleEnvironment[SokobanEngine]):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        self.engine: SokobanEngine = SokobanEngine(task_instance)

    # ------------------------------------------------------------------ #
    # lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:
        # no special engine cleanup needed beyond GC, return final board
        priv, pub = await self.engine._serialize_engine(), None  # placeholder
        obs_dict = {"terminated": True, "message": "Environment terminated."}
        return obs_dict  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # main API                                                            #
    # ------------------------------------------------------------------ #

    def validate_tool_calls(self, tool_calls: List[List[EnvToolCall]]) -> None:
        if not tool_calls or not isinstance(tool_calls[0][0], EnvToolCall):
            raise ValueError("tool_calls must be a nested list of EnvToolCall objects")

    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:
        self.validate_tool_calls(tool_calls)
        # assume first inner list contains one call with attribute `action`
        action: int = tool_calls[0][0].action  # type: ignore[attr-defined]
        priv, pub = await self.engine._step_engine(action)
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:
        # create a final observation callable if none provided
        obs_cb = (
            self.custom_checkpoint_observation_callable
            or SynthSokobanObservationCallable()
        )
        priv_state = SokobanPrivateState(
            reward_last=0, total_reward=0, terminated=True, truncated=False
        )
        pub_state = SokobanPublicState(**{})  # fill from engine if needed
        return await obs_cb.get_observation(pub_state, priv_state)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #

    async def _to_observation(
        self,
        priv: SokobanPrivateState,
        pub: SokobanPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (obs_cb or SynthSokobanObservationCallable()).get_observation(
            pub, priv
        )

    # ------------------------------------------------------------------ #
    # reproducibility passthrough                                         #
    # ------------------------------------------------------------------ #

    async def _serialize_engine(self) -> Any:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, snapshot: Any) -> "SokobanEnvironment":
        eng = await SokobanEngine._deserialize_engine(snapshot)
        env = cls(eng.task_instance)
        env.engine = eng
        return env
