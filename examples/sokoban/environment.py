from typing import List, Optional, Any, Dict
from examples.sokoban.engine import (
    SokobanEngine,
    SynthSokobanObservationCallable,
    SokobanPrivateState,
    SokobanPublicState,
    SynthSokobanCheckpointObservationCallable,
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
        self.name = "Sokoban"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        self.engine: SokobanEngine = SokobanEngine(task_instance)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:
        priv, pub = await self.engine._serialize_engine(), None  # placeholder
        obs_dict = {"terminated": True, "message": "Environment terminated."}
        return obs_dict  # type: ignore[return-value]

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
        # Construct public state from the engine's wrapped env
        pub_state = SokobanPublicState(
            dim_room=self.engine.package_sokoban_env.dim_room,
            room_fixed=self.engine.package_sokoban_env.room_fixed.copy(),
            room_state=self.engine.package_sokoban_env.room_state.copy(),
            player_position=tuple(self.engine.package_sokoban_env.player_position),
            boxes_on_target=self.engine.package_sokoban_env.boxes_on_target,
            num_steps=self.engine.package_sokoban_env.num_env_steps,
            max_steps=self.engine.package_sokoban_env.max_steps,
            last_action_name="",  # not tracked for checkpoint
            num_boxes=self.engine.package_sokoban_env.num_boxes,
        )

        # Construct private state
        # Determine terminated and truncated status from the engine's package_sokoban_env
        terminated = bool(
            self.engine.package_sokoban_env.boxes_on_target
            == self.engine.package_sokoban_env.num_boxes
        )
        truncated = bool(
            self.engine.package_sokoban_env.num_env_steps
            >= self.engine.package_sokoban_env.max_steps
        )
        # Reward_last for a checkpoint could be considered 0 or the last step's reward.
        # For a final summary, total_reward is key. Let's assume reward_last is not critical for checkpoint.
        priv_state = SokobanPrivateState(
            reward_last=self.engine.package_sokoban_env.reward_last,  # Use last actual reward
            total_reward=self.engine._total_reward,  # type: ignore[attr-defined]
            terminated=terminated,
            truncated=truncated,
        )

        # Use SynthSokobanCheckpointObservationCallable by default
        obs_cb = (
            self.custom_checkpoint_observation_callable
            or SynthSokobanCheckpointObservationCallable()  # Changed default
        )
        return await obs_cb.get_observation(pub_state, priv_state)
    
    async def _to_observation(
        self,
        priv: SokobanPrivateState,
        pub: SokobanPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (obs_cb or SynthSokobanObservationCallable()).get_observation(
            pub, priv
        )

    async def _serialize_engine(self) -> Any:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, snapshot: Any) -> "SokobanEnvironment":
        eng = await SokobanEngine._deserialize_engine(snapshot)
        env = cls(eng.task_instance)
        env.engine = eng
        return env
