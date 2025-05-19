"""CrafterClassicEnvironment — thin wrapper exposing CrafterEngine via StatefulEnvironment API."""

from __future__ import annotations

from typing import List, Optional, Any

from examples.crafter_classic.engine import (
    CrafterEngine,
    CrafterObservationCallable,
    CrafterPrivateState,
    CrafterPublicState,
)
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.reproducibility.core import ReproducibleEnvironment
from src.stateful.core import StatefulEnvironment
from src.tasks.core import TaskInstance
from src.environment.tools import EnvToolCall


class CrafterClassicEnvironment(
    StatefulEnvironment, ReproducibleEnvironment[CrafterEngine]
):
    """Environment wrapper bridging agent tool‑calls to `crafter.Env` dynamics."""

    def __init__(
        self,
        task_instance: TaskInstance,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.name = "CrafterClassic"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs
        self.custom_checkpoint_observation_callable = custom_ckpt_obs
        self.engine: CrafterEngine = CrafterEngine(task_instance)

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle helpers
    # ────────────────────────────────────────────────────────────────────

    async def initialize(self) -> InternalObservation:  # type: ignore[override]
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:  # type: ignore[override]
        # No engine‑side cleanup needed; just emit marker.
        return {"terminated": True, "message": "Environment terminated."}  # type: ignore[return-value]

    # ────────────────────────────────────────────────────────────────────
    # Step + checkpoint
    # ────────────────────────────────────────────────────────────────────

    def validate_tool_calls(self, tool_calls: List[List[EnvToolCall]]) -> None:
        if not tool_calls or not isinstance(tool_calls[0][0], EnvToolCall):
            raise ValueError("tool_calls must be a nested list of EnvToolCall objects")

    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:  # type: ignore[override]
        self.validate_tool_calls(tool_calls)
        action_raw = tool_calls[0][0].action  # type: ignore[attr-defined]
        action = self.engine._validate_action_engine(action_raw)  # ensure legal
        priv, pub = await self.engine._step_engine(action)
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:  # type: ignore[override]
        # Build snapshot‑style states directly from live engine.
        player = self.engine.env._player  # type: ignore[attr-defined]
        pub_state = CrafterPublicState(
            inventory=player.inventory.copy(),
            achievements_status={k: v > 0 for k, v in player.achievements.items()},
            player_position=tuple(player.pos),
            player_direction=player.facing,
            semantic_map=getattr(self.engine.env, "_sem_view", lambda: None)(),
            world_material_map=self.engine.env._world._mat_map.copy(),  # type: ignore[attr-defined]
            observation_image=self.engine.env.render(),
            num_steps_taken=self.engine.env._step,  # type: ignore[attr-defined]
            max_steps_episode=self.engine.env._length,  # type: ignore[attr-defined]
        )

        priv_state = CrafterPrivateState(
            reward_last_step=0.0,
            total_reward_episode=self.engine._total_reward,  # type: ignore[attr-defined]
            achievements_current_values=player.achievements.copy(),
            terminated=player.health <= 0,
            truncated=self.engine.env._step >= self.engine.env._length,  # type: ignore[attr-defined]
            player_internal_stats={
                "health": player.health,
                "food": player.inventory.get("food"),
                "drink": player.inventory.get("drink"),
                "energy": player.inventory.get("energy"),
            },
            world_rng_state_snapshot=self.engine.env._world.random.getstate(),  # type: ignore[attr-defined]
        )

        obs_cb = (
            self.custom_checkpoint_observation_callable or CrafterObservationCallable()
        )
        return await obs_cb.get_observation(pub_state, priv_state)

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    async def _to_observation(
        self,
        priv: CrafterPrivateState,
        pub: CrafterPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        return await (obs_cb or CrafterObservationCallable()).get_observation(pub, priv)

    # ────────────────────────────────────────────────────────────────────
    # ReproducibleEnvironment plumbing
    # ────────────────────────────────────────────────────────────────────

    async def _serialize_engine(self) -> Any:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(cls, snapshot: Any) -> "CrafterClassicEnvironment":
        eng = await CrafterEngine._deserialize_engine(snapshot)
        env = cls(eng.task_instance)
        env.engine = eng
        return env
