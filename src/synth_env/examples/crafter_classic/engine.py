"""CrafterEngine — Stateful, reproducible wrapper around danijar/crafter.Env.
This file follows the same structure as the SokobanEngine shown earlier.
"""

from __future__ import annotations

# Import logging configuration first to suppress JAX debug messages
from .config_logging import safe_compare


import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import crafter  # type: ignore
import copy
import dataclasses

from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_env.tasks.core import TaskInstance
from synth_env.reproducibility.core import IReproducibleEngine
from synth_env.environment.rewards.core import RewardStack, RewardComponent  # Added

# Local helper imports (must exist relative to this file)
from .engine_helpers.action_map import CRAFTER_ACTION_MAP  # action‑name → int
from .engine_helpers.serialization import (
    serialize_world_object,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Dataclasses for snapshot & (public, private) runtime state
# -----------------------------------------------------------------------------


@dataclass
class CrafterEngineSnapshot(StatefulEngineSnapshot):
    env_raw_state: Any  # from crafter.Env.save()
    total_reward_snapshot: float
    crafter_seed: Optional[int] = None
    # Store previous states needed for reward calculation to resume correctly
    previous_public_state_snapshot: Optional[Dict] = None
    previous_private_state_snapshot: Optional[Dict] = None
    # Add _previous_public_state_for_reward and _previous_private_state_for_reward if needed for perfect resume
    # For RewardStack, its configuration is fixed at init. If it had internal state, that would need saving.


@dataclass
class CrafterPublicState:
    inventory: Dict[str, int]
    achievements_status: Dict[str, bool]
    player_position: Tuple[int, int]
    player_direction: Union[int, Tuple[int, int]]
    semantic_map: Optional[np.ndarray]
    world_material_map: np.ndarray
    observation_image: np.ndarray
    num_steps_taken: int
    max_steps_episode: int
    error_info: Optional[str] = None

    def diff(self, prev_state: "CrafterPublicState") -> Dict[str, Any]:
        changes = {}
        for field in self.__dataclass_fields__:  # type: ignore[attr-defined]
            new_v, old_v = getattr(self, field), getattr(prev_state, field)
            if isinstance(new_v, np.ndarray):
                if not np.array_equal(new_v, old_v):
                    changes[field] = True
            elif new_v != old_v:
                changes[field] = (old_v, new_v)
        return changes


@dataclass
class CrafterPrivateState:
    reward_last_step: float
    total_reward_episode: float
    achievements_current_values: Dict[str, int]
    terminated: bool
    truncated: bool
    player_internal_stats: Dict[str, Any]
    world_rng_state_snapshot: Any

    def diff(self, prev_state: "CrafterPrivateState") -> Dict[str, Any]:
        changes = {}
        for field in self.__dataclass_fields__:  # type: ignore[attr-defined]
            new_v, old_v = getattr(self, field), getattr(prev_state, field)
            if new_v != old_v:
                changes[field] = (old_v, new_v)
        return changes


# -----------------------------------------------------------------------------
# Observation helpers
# -----------------------------------------------------------------------------


class CrafterObservationCallable(GetObservationCallable):
    def __init__(self) -> None:
        pass

    async def get_observation(
        self, pub: CrafterPublicState, priv: CrafterPrivateState
    ) -> InternalObservation:  # type: ignore[override]
        observation: Dict[str, Any] = {
            "inventory": pub.inventory,
            "achievements": pub.achievements_status,
            "player_pos": pub.player_position,
            "steps": pub.num_steps_taken,
            "reward_last": priv.reward_last_step,
            "total_reward": priv.total_reward_episode,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }
        return observation  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# CrafterEngine implementation
# -----------------------------------------------------------------------------


class CrafterEngine(StatefulEngine, IReproducibleEngine):
    """StatefulEngine wrapper around `crafter.Env` supporting full snapshotting."""

    task_instance: TaskInstance
    env: crafter.Env

    # ────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self._total_reward: float = 0.0
        self._current_action_for_reward: Optional[int] = None
        self._previous_public_state_for_reward: Optional[CrafterPublicState] = None
        self._previous_private_state_for_reward: Optional[CrafterPrivateState] = (
            None  # For stat changes
        )

        # Initialize achievements tracking
        self.achievements_unlocked: set = set()

        cfg = getattr(task_instance, "config", {}) or {}
        area: Tuple[int, int] = tuple(cfg.get("area", (64, 64)))  # type: ignore[arg-type]
        length: int = int(cfg.get("length", 10000))

        # Get seed from metadata if available, otherwise fall back to config
        seed: Optional[int] = cfg.get("seed")
        if hasattr(task_instance, "metadata") and hasattr(
            task_instance.metadata, "seed"
        ):
            seed = task_instance.metadata.seed

        self.env = crafter.Env(area=area, length=length, seed=seed)
        # store original seed for reproducibility
        self.env._seed = seed

        self.reward_stack = RewardStack(
            components=[
                CrafterAchievementComponent(),
                CrafterPlayerStatComponent(),
                CrafterStepPenaltyComponent(penalty=-0.001),
            ]
        )

    # ────────────────────────────────────────────────────────────────────────
    # Utility: action validation / mapping
    # ────────────────────────────────────────────────────────────────────────

    def _validate_action_engine(self, action: Union[int, str]) -> int:  # type: ignore[override]
        if isinstance(action, str):
            action = CRAFTER_ACTION_MAP.get(action, 0)
        if not isinstance(action, int):
            return 0
        return int(np.clip(action, 0, len(crafter.constants.actions) - 1))  # type: ignore

    # ────────────────────────────────────────────────────────────────────────
    # Core StatefulEngine API
    # ────────────────────────────────────────────────────────────────────────

    async def _reset_engine(
        self, *, seed: Optional[int] | None = None
    ) -> Tuple[CrafterPrivateState, CrafterPublicState]:
        if seed is not None:
            # Re‑instantiate env with new seed to match crafter's internal reseeding convention
            self.env = crafter.Env(
                area=self.env._area, length=self.env._length, seed=seed
            )
        obs_img = self.env.reset()
        self._total_reward = 0.0
        pub = self._build_public_state(obs_img)
        priv = self._build_private_state(reward=0.0, terminated=False, truncated=False)
        return priv, pub

    async def _step_engine(
        self, action: int
    ) -> Tuple[CrafterPrivateState, CrafterPublicState]:
        try:
            # Validate action is in valid range
            if action < 0 or action >= self.env.action_space.n:
                raise ValueError(
                    f"Invalid action {action}, must be in range [0, {self.env.action_space.n})"
                )

            current_pub_state = self._build_public_state(self.env.render())

            # Step the environment
            obs, reward, done, info = self.env.step(action)

            # Update internal state
            self.obs = obs
            self.done = done
            self.info = info
            self.last_reward = reward

            # Step count is tracked by the crafter environment itself in self.env._step

            # Process achievements - check what was unlocked this step
            new_achievements = set()
            if "achievements" in info:
                for achievement, status in info["achievements"].items():
                    if status and achievement not in self.achievements_unlocked:
                        new_achievements.add(achievement)
                        self.achievements_unlocked.add(achievement)

            # Calculate reward
            reward_from_stack = 0
            try:
                if hasattr(self, "_reward_stack") and self._reward_stack:
                    reward_from_stack = sum(self._reward_stack)
                    self._reward_stack.clear()
            except Exception as e:
                reward_from_stack = 0

            # Create private state
            # Current episode reward
            final_reward = self._total_reward + reward + reward_from_stack
            self._total_reward = final_reward

            # Determine proper termination reason based on game state
            player = self.env._player  # type: ignore[attr-defined]
            current_step = self.env._step  # type: ignore[attr-defined]
            max_steps = self.env._length  # type: ignore[attr-defined]

            # Check if player died (health <= 0)
            player_died = player.health <= 0

            # Check if max steps reached
            max_steps_reached = current_step >= max_steps

            # Set termination flags properly:
            # - terminated=True only if player actually died
            # - truncated=True only if episode ended due to step limit
            if done:
                if player_died:
                    terminated = True
                    truncated = False
                elif max_steps_reached:
                    terminated = False
                    truncated = True
                else:
                    # Fallback: if done=True but unclear reason, assume timeout
                    terminated = False
                    truncated = True
            else:
                terminated = False
                truncated = False

            final_priv_state = self._build_private_state(
                final_reward, terminated, truncated
            )

            self._previous_public_state_for_reward = current_pub_state
            self._previous_private_state_for_reward = final_priv_state

            return final_priv_state, current_pub_state

        except Exception as e:
            # Create error state
            error_pub_state = self._get_public_state_from_env()
            error_pub_state.error_info = f"Step engine error: {e}"
            error_priv_state = self._get_private_state_from_env(
                reward=-1.0, terminated=True, truncated=False
            )
            return error_priv_state, error_pub_state

    # ------------------------------------------------------------------
    # Rendering (simple text summary)
    # ------------------------------------------------------------------

    async def _render(
        self,
        private_state: CrafterPrivateState,
        public_state: CrafterPublicState,
        get_observation: Optional[GetObservationCallable] = None,
    ) -> str:  # type: ignore[override]
        obs_cb = get_observation or CrafterObservationCallable()
        obs = await obs_cb.get_observation(public_state, private_state)
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            header = f"steps: {public_state.num_steps_taken}/{public_state.max_steps_episode} | "
            header += f"last_r: {private_state.reward_last_step:.2f} | total_r: {private_state.total_reward_episode:.2f}"
            inv = ", ".join(f"{k}:{v}" for k, v in public_state.inventory.items() if v)
            ach = ", ".join(k for k, v in public_state.achievements_status.items() if v)
            return f"{header}\ninv: {inv}\nach: {ach}"
        return str(obs)

    # ------------------------------------------------------------------
    # Snapshotting for exact reproducibility
    # ------------------------------------------------------------------

    async def _serialize_engine(self) -> CrafterEngineSnapshot:
        world = self.env._world  # type: ignore[attr-defined]
        objects_state = [
            None if o is None else serialize_world_object(o) for o in world._objects
        ]
        # capture total reward and original seed
        total_reward = self._total_reward
        snap = CrafterEngineSnapshot(
            env_raw_state=self.env.save(),
            total_reward_snapshot=total_reward,
            crafter_seed=self.env._seed,
            previous_public_state_snapshot=dataclasses.asdict(
                self._previous_public_state_for_reward
            )
            if self._previous_public_state_for_reward
            else None,
            previous_private_state_snapshot=dataclasses.asdict(
                self._previous_private_state_for_reward
            )
            if self._previous_private_state_for_reward
            else None,
        )
        return snap

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: CrafterEngineSnapshot, task_instance: TaskInstance
    ) -> "CrafterEngine":
        engine = cls(task_instance)
        engine.env.load(snapshot.env_raw_state)
        engine._total_reward = snapshot.total_reward_snapshot
        engine.env._seed = snapshot.crafter_seed
        _ = engine.env.reset()  # create initial world structure
        # Re-establish previous states for reward system continuity if first step after load
        engine._previous_public_state_for_reward = engine._build_public_state(
            engine.env.render()
        )
        # Safe comparisons to avoid string vs int errors
        health_dead = safe_compare(0, engine.env._player.health, ">=")
        step_exceeded = safe_compare(engine.env._length, engine.env._step, "<=")
        engine._previous_private_state_for_reward = engine._build_private_state(
            0.0, health_dead, step_exceeded
        )
        return engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_public_state(
        self, obs_img: np.ndarray, info: Optional[Dict[str, Any]] | None = None
    ) -> CrafterPublicState:
        try:
            if info is None:
                player = self.env._player  # type: ignore[attr-defined]
                # Safe achievement status check
                achievements_status = {}
                for k, v in player.achievements.items():
                    achievements_status[k] = safe_compare(0, v, "<")
                inventory = player.inventory.copy()
                semantic = getattr(self.env, "_sem_view", lambda: None)()
            else:
                inventory = info.get("inventory", {})
                # Safe achievement status check from info
                achievements_status = {}
                achievements_info = info.get("achievements", {})
                for k, v in achievements_info.items():
                    achievements_status[k] = safe_compare(0, v, "<")
                semantic = info.get("semantic")

            player = self.env._player  # type: ignore[attr-defined]
            return CrafterPublicState(
                inventory=inventory,
                achievements_status=achievements_status,
                player_position=tuple(player.pos),  # type: ignore[attr-defined]
                player_direction=player.facing,  # type: ignore[attr-defined]
                semantic_map=semantic,
                world_material_map=self.env._world._mat_map.copy(),  # type: ignore[attr-defined]
                observation_image=obs_img,
                num_steps_taken=self.env._step,  # type: ignore[attr-defined]
                max_steps_episode=self.env._length,  # type: ignore[attr-defined]
                error_info=info.get("error_info") if info else None,
            )
        except Exception as e:
            logging.error(f"Error building public state: {e}")
            # Return minimal safe state
            return CrafterPublicState(
                inventory={},
                achievements_status={},
                player_position=(0, 0),
                player_direction=0,
                semantic_map=None,
                world_material_map=np.zeros((1, 1), dtype=np.uint8),
                observation_image=obs_img
                if obs_img is not None
                else np.zeros((64, 64, 3), dtype=np.uint8),
                num_steps_taken=0,
                max_steps_episode=10000,
                error_info=f"State building error: {e}",
            )

    def _build_private_state(
        self, reward: float, terminated: bool, truncated: bool
    ) -> CrafterPrivateState:
        player = self.env._player  # type: ignore[attr-defined]
        stats = {
            "health": player.health,
            "food": player.inventory.get("food"),
            "drink": player.inventory.get("drink"),
            "energy": player.inventory.get("energy"),
            "_hunger": getattr(player, "_hunger", 0),
            "_thirst": getattr(player, "_thirst", 0),
        }
        return CrafterPrivateState(
            reward_last_step=reward,
            total_reward_episode=self._total_reward,
            achievements_current_values=player.achievements.copy(),
            terminated=terminated,
            truncated=truncated,
            player_internal_stats=stats,
            world_rng_state_snapshot=self.env._world.random.get_state(),  # type: ignore[attr-defined]
        )

    def _get_public_state_from_env(self) -> CrafterPublicState:
        """Helper method to get current public state from synth_env.environment"""
        try:
            obs_img = self.env.render()
            return self._build_public_state(obs_img)
        except Exception as e:
            logging.error(f"Error getting public state from env: {e}")
            # Return default state
            return CrafterPublicState(
                inventory={},
                achievements_status={},
                player_position=(0, 0),
                player_direction=0,
                semantic_map=None,
                world_material_map=np.zeros((1, 1), dtype=np.uint8),
                observation_image=np.zeros((64, 64, 3), dtype=np.uint8),
                num_steps_taken=0,
                max_steps_episode=10000,
                error_info=f"State extraction error: {e}",
            )

    def _get_private_state_from_env(
        self, reward: float, terminated: bool, truncated: bool
    ) -> CrafterPrivateState:
        """Helper method to get current private state from synth_env.environment"""
        try:
            return self._build_private_state(reward, terminated, truncated)
        except Exception as e:
            logging.error(f"Error getting private state from env: {e}")
            # Return default state
            return CrafterPrivateState(
                reward_last_step=reward,
                total_reward_episode=0.0,
                achievements_current_values={},
                terminated=terminated,
                truncated=truncated,
                player_internal_stats={},
                world_rng_state_snapshot=None,
            )


# --- Reward Components ---
class CrafterAchievementComponent(RewardComponent):
    async def score(self, state: CrafterPublicState, action: Dict[str, Any]) -> float:
        prev_achievements = action.get("previous_public_state_achievements", {})
        current_achievements = state.achievements_status
        new_achievements = sum(
            1
            for ach, status in current_achievements.items()
            if status and not prev_achievements.get(ach)
        )
        return float(new_achievements) * 0.1


class CrafterPlayerStatComponent(RewardComponent):
    async def score(self, state: CrafterPrivateState, action: Dict[str, Any]) -> float:
        current_health = state.player_internal_stats.get("health", 0)
        prev_health = action.get("previous_private_state_stats", {}).get(
            "health", current_health
        )
        if current_health < prev_health:
            return -0.05  # Lost health penalty
        return 0.0


class CrafterStepPenaltyComponent(RewardComponent):
    def __init__(self, penalty: float = -0.001):
        super().__init__()
        self.penalty = penalty
        self.weight = 1.0

    async def score(self, state: Any, action: Any) -> float:
        return self.penalty
