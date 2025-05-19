"""CrafterEngine — Stateful, reproducible wrapper around danijar/crafter.Env.
This file follows the same structure as the SokobanEngine shown earlier.
"""

from __future__ import annotations

import examples.crafter_classic.engine_deterministic_patch # Apply patch

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import crafter  # type: ignore
import collections

from src.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.tasks.core import TaskInstance
from src.reproducibility.core import IReproducibleEngine

# Local helper imports (must exist relative to this file)
from .engine_helpers.action_map import CRAFTER_ACTION_MAP  # action‑name → int
from .engine_helpers.serialization import (
    serialize_world_object,
    deserialize_world_object,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Dataclasses for snapshot & (public, private) runtime state
# -----------------------------------------------------------------------------


@dataclass
class CrafterEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict[str, Any]
    # ── static env config ────────────────────────────────────────────────────
    env_area: Tuple[int, int]
    env_length: int
    env_initial_seed: Optional[int]
    # ── dynamic episode / player status ─────────────────────────────────────
    current_episode: int
    current_step: int
    last_health: float
    unlocked_achievements_in_episode: List[str]
    total_reward_episode: float  # cumulative reward to restore engine._total_reward
    # ── world state ─────────────────────────────────────────────────────────
    world_mat_map: List[List[int]]
    world_obj_map: List[List[int]]
    world_objects_state: List[Optional[Dict[str, Any]]]
    world_daylight: float
    world_rng_state: Any  # State from numpy.random.RandomState.get_state()


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
        return {
            "inventory": pub.inventory,
            "achievements": pub.achievements_status,
            "player_pos": pub.player_position,
            "steps": pub.num_steps_taken,
            "reward_last": priv.reward_last_step,
            "total_reward": priv.total_reward_episode,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }


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

        cfg = getattr(task_instance, "config", {}) or {}
        area: Tuple[int, int] = tuple(cfg.get("area", (64, 64)))  # type: ignore[arg-type]
        length: int = int(cfg.get("length", 10000))
        seed: Optional[int] = cfg.get("seed")

        self.env = crafter.Env(area=area, length=length, seed=seed)
        # store original seed for reproducibility
        self.env._seed = seed

    # ────────────────────────────────────────────────────────────────────────
    # Utility: action validation / mapping
    # ────────────────────────────────────────────────────────────────────────

    def _validate_action_engine(self, action: Union[int, str]) -> int:  # type: ignore[override]
        if isinstance(action, str):
            action = CRAFTER_ACTION_MAP.get(action, 0)
        if not isinstance(action, int):
            return 0
        return int(np.clip(action, 0, len(crafter.constants.actions) - 1))

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
        obs_img, reward, done, info = self.env.step(action)
        self._total_reward += reward
        terminated = self.env._player.health <= 0  # type: ignore[attr-defined]
        truncated = done and not terminated
        pub = self._build_public_state(obs_img, info)
        priv = self._build_private_state(reward, terminated, truncated)
        return priv, pub

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
            None if o is None else serialize_world_object(o)
            for o in world._objects
        ]
        # capture total reward and original seed
        total_reward = self._total_reward
        snap = CrafterEngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            env_area=tuple(self.env._area),  # type: ignore[attr-defined]
            env_length=self.env._length,  # type: ignore[attr-defined]
            env_initial_seed=self.env._seed,  # stored original seed
            current_episode=self.env._episode,  # type: ignore[attr-defined]
            current_step=self.env._step,  # type: ignore[attr-defined]
            last_health=self.env._last_health,  # type: ignore[attr-defined]
            total_reward_episode=total_reward,
            unlocked_achievements_in_episode=list(self.env._unlocked),  # type: ignore[attr-defined]
            world_mat_map=world._mat_map.tolist(),
            world_obj_map=world._obj_map.tolist(),
            world_objects_state=objects_state,
            world_daylight=world.daylight,
            world_rng_state=world.random.get_state(),
        )
        return snap

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: "CrafterEngineSnapshot"
    ) -> "CrafterEngine":
        task_instance = await TaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls.__new__(cls)
        StatefulEngine.__init__(engine)
        engine.task_instance = task_instance
        # reconstruct env
        engine.env = crafter.Env(
            area=snapshot.env_area,
            length=snapshot.env_length,
            seed=snapshot.env_initial_seed,
        )
        # restore original seed attribute
        engine.env._seed = snapshot.env_initial_seed
        _ = engine.env.reset()  # create initial world structure
        # ── restore high-level counters ────────────────────────────────
        engine.env._episode = snapshot.current_episode  # type: ignore[attr-defined]
        engine.env._step = snapshot.current_step  # type: ignore[attr-defined]
        engine.env._last_health = snapshot.last_health  # type: ignore[attr-defined]
        engine.env._unlocked = set(snapshot.unlocked_achievements_in_episode)  # type: ignore[attr-defined]
        # restore total reward
        engine._total_reward = snapshot.total_reward_episode
        # ── restore world ──────────────────────────────────────────────
        world = engine.env._world  # type: ignore[attr-defined]
        # reinitialize world RNG and structure
        engine.env._world.reset(seed=hash((snapshot.env_initial_seed, snapshot.current_episode)) % (2 ** 31 - 1))
        # ensure future additions have unique indices
        engine.env._world._next_id = len(snapshot.world_objects_state) + 1
        # restore world maps and state
        world._mat_map = np.array(snapshot.world_mat_map, dtype=int)
        world._obj_map = np.array(snapshot.world_obj_map, dtype=int)
        world.daylight = snapshot.world_daylight
        world.random.set_state(snapshot.world_rng_state)
        # rebuild objects list and chunk index exactly
        objects: List[Any] = []
        world._chunks = collections.defaultdict(set)
        for obj_state in snapshot.world_objects_state:
            if obj_state is None:
                objects.append(None)
            else:
                obj = deserialize_world_object(obj_state, world)
                objects.append(obj)
                world._chunks[world.chunk_key(obj.pos)].add(obj)
        world._objects = objects
        # find player instance
        for obj in objects:
            if obj is not None and isinstance(obj, crafter.objects.Player):  # type: ignore[attr-defined]
                engine.env._player = obj  # type: ignore[attr-defined]
                break
        # assign player reference for entities that need it
        for obj in objects:
            if obj is not None and (
                isinstance(obj, crafter.objects.Zombie) or isinstance(obj, crafter.objects.Skeleton)
            ):  # type: ignore[attr-defined]
                obj.player = engine.env._player  # type: ignore[attr-defined]
        return engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_public_state(
        self, obs_img: np.ndarray, info: Optional[Dict[str, Any]] | None = None
    ) -> CrafterPublicState:
        if info is None:
            player = self.env._player  # type: ignore[attr-defined]
            achievements_status = {k: v > 0 for k, v in player.achievements.items()}
            inventory = player.inventory.copy()
            semantic = getattr(self.env, "_sem_view", lambda: None)()
        else:
            inventory = info.get("inventory", {})
            achievements_status = {
                k: v > 0 for k, v in info.get("achievements", {}).items()
            }
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
