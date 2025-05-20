"""NMMO 3 Environment wrapper.

Bridges tool-calls coming from the agent → NMMO3Engine dynamics, and turns
the engine's Public/Private dataclasses into an `InternalObservation`
dict ready for the evaluation pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict

from examples.nmmo_classic.engine import (
    NMMO3Engine,
    NMMO3ObservationCallable,
    NMMO3PublicState,
    NMMO3PrivateState,
)
from environment.shared_engine import GetObservationCallable, InternalObservation
from stateful.core import StatefulEnvironment
from reproducibility.core import ReproducibleEnvironment
from environment.tools import EnvToolCall
from tasks.core import TaskInstance


class NMMO3Environment(StatefulEnvironment, ReproducibleEnvironment[NMMO3Engine]):
    """
    Exactly the same responsibilities as CrafterClassicEnvironment / SokobanEnvironment:

        • hold a live Engine instance
        • convert tool-calls → validated env actions
        • expose .initialize / .step / .checkpoint / .terminate
        • forward (de)serialization to the underlying engine
    """

    # ------------------------------------------------------------------ #
    # ctor / lifecycle helpers                                           #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "NMMO3"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        self.engine: NMMO3Engine = NMMO3Engine(task_instance)

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

        # Convention: the *first* tool-call carries the NMMO action dict
        # Users may send either the raw Dict[str, gym.Space.sample] structure
        # or a pre-flattened vector that the policy uses internally – we just
        # forward whatever `.action` contains.  Validation is delegated to
        # NMMO3Engine (and ultimately to nmmo itself).
        action_payload: Dict[str, Any] = tool_calls[0][0].action  # type: ignore[attr-defined]

        priv, pub = await self.engine._step_engine(action_payload)
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

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
        priv: NMMO3PrivateState,
        pub: NMMO3PublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (obs_cb or NMMO3ObservationCallable()).get_observation(pub, priv)

    # ------------------------------------------------------------------ #
    # ReproducibleEnvironment plumbing                                   #
    # ------------------------------------------------------------------ #
    async def _serialize_engine(self) -> Any:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, snapshot: Any) -> "NMMO3Environment":
        eng = await NMMO3Engine._deserialize_engine(snapshot)
        env = cls(eng.task_instance)
        env.engine = eng
        return env
