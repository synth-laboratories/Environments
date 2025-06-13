"""Neural MMO Classic Engine

NeuralMMOEngine — stateful, reproducible wrapper around Neural MMO environment.
Design goals:
  • Single-agent control with multi-agent environment support
  • Dual observation modes: image and vector-based
  • Full reproducibility via state snapshots
  • Clean integration with PufferLib for multi-agent handling
"""

from __future__ import annotations

import pickle
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import nmmo
from pufferlib.emulation import PettingZooPufferEnv

from synth_env.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_env.reproducibility.core import IReproducibleEngine
from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.tasks.core import TaskInstance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ════════════════════════════════════════════════════════════════════════════
#  Snapshot + runtime state dataclasses
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class NeuralMMOEngineSnapshot(StatefulEngineSnapshot):
    """Binary snapshot – stores the *entire* PufferEnv via pickle plus engine state."""

    task_instance_dict: Dict[str, Any]
    pickle_blob: bytes
    total_reward: float
    controlled_agent_id: int
    rng_state: Dict[str, Any]  # NumPy RNG state for reproducibility


@dataclass
class NeuralMMOPublicState:
    # ------- episode status --------------------------------------------------
    tick: int
    num_steps_taken: int
    max_episode_steps: int
    # ------- ego stats -------------------------------------------------------
    agent_id: int
    position: Tuple[int, int]
    facing: int
    health: int
    stamina: int
    inventory: Dict[str, int]
    # ------- observations ----------------------------------------------------
    local_terrain: np.ndarray  # (H, W)
    visible_entities: np.ndarray  # structured NumPy view, not pandas
    # ------- scoring ---------------------------------------------------------
    team_score: float
    personal_score: float

    # ---- diff helper --------------------------------------------------------
    def diff(self, prev: "NeuralMMOPublicState") -> Dict[str, Any]:
        changed: Dict[str, Any] = {}
        for f in self.__dataclass_fields__:
            a, b = getattr(self, f), getattr(prev, f)
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    changed[f] = True
            elif a != b:
                changed[f] = (b, a)
        return changed


@dataclass
class SkillSnapshot:
    lvl: int  # current level
    xp: int  # accumulated XP


@dataclass
class NeuralMMOPrivateState:
    # ─── rewards / termination ──────────────────────────────────────────────
    reward_last_step: float
    total_reward_episode: float
    terminated: bool
    truncated: bool

    # ─── reproducibility ────────────────────────────────────────────────────
    env_rng_state_snapshot: Dict[str, Any]

    # ─── progression & debugging hooks ─────────────────────────────────────
    # 1. Per-skill progress (8 skills in default config)
    skills: Dict[str, SkillSnapshot] = field(default_factory=dict)
    # 2. Task / predicate flags (if you enabled them in cfg.Tasks)
    achievements_status: Dict[str, bool] = field(default_factory=dict)
    # 3. Any other low-level stats you find useful (cool-downs, buffs, etc.)
    agent_internal_stats: Dict[str, Any] = field(default_factory=dict)

    # optional helper for diffing two states (handy in tests)
    def diff(self, prev: "NeuralMMOPrivateState") -> Dict[str, Any]:
        changed: Dict[str, Any] = {}
        for f in self.__dataclass_fields__:
            new, old = getattr(self, f), getattr(prev, f)
            if new != old:
                changed[f] = (old, new)
        return changed


# ════════════════════════════════════════════════════════════════════════════
#  Observation helper
# ════════════════════════════════════════════════════════════════════════════


class NeuralMMOObservationCallable(GetObservationCallable):
    """Observation transformer supporting both image and vector modes."""

    def __init__(self, engine: Optional["NeuralMMOEngine"] = None):
        self.engine = engine

    async def get_observation(  # type: ignore[override]
        self, pub: NeuralMMOPublicState, priv: NeuralMMOPrivateState
    ) -> InternalObservation:
        base_obs = {
            "tick": pub.tick,
            "position": pub.position,
            "health": pub.health,
            "stamina": pub.stamina,
            "inventory": pub.inventory,
            "terrain": pub.local_terrain,
            "visible_entities": pub.visible_entities,
            "reward_last": priv.reward_last_step,
            "total_reward": priv.total_reward_episode,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }

        # Add image observation if in image mode
        if self.engine and self.engine._observation_mode == "image":
            base_obs["image"] = self.engine._render_image(pub)

        return base_obs


# ════════════════════════════════════════════════════════════════════════════
#  Engine implementation
# ════════════════════════════════════════════════════════════════════════════


class NeuralMMOEngine(StatefulEngine, IReproducibleEngine):
    """Neural MMO engine with dual observation modes (image/vector) and multi-agent support."""

    task_instance: TaskInstance
    env: Union[nmmo.Env, PettingZooPufferEnv]
    _puffer_env: Optional[PettingZooPufferEnv]
    _total_reward: float
    _controlled_agent: int
    _observation_mode: str  # "image" or "vector"
    _map_data: Optional[np.ndarray]
    # Last known states for checkpointing
    _last_public_state: NeuralMMOPublicState | None
    _last_private_state: NeuralMMOPrivateState | None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, task_instance: TaskInstance, observation_mode: str = "vector"):
        self.task_instance = task_instance
        self._total_reward = 0.0
        self._observation_mode = observation_mode
        self._puffer_env = None
        self._map_data = None
        # Initialize last states
        self._last_public_state = None
        self._last_private_state = None

        # --- build env ----------------------------------------------------
        # Create nmmo config from task metadata
        import nmmo.core.config as config

        # Create a custom config class with our settings
        meta = task_instance.metadata
        map_size = getattr(meta, "map_size", 128)
        seed = getattr(meta, "seed", 0)

        class TestConfig(config.Default):
            MAP_BORDER = 10  # Must be > PLAYER_VISION_RADIUS
            MAP_SIZE = map_size
            PLAYER_VISION_RADIUS = 8
            TICK_LIMIT = 1000
            SEED = seed
            NPC_SYSTEM_ENABLED = False  # Disable NPCs to avoid spawning bugs

        nmmo_config = TestConfig()

        # Create the base NMMO environment
        self.env = nmmo.Env(config=nmmo_config)

        # Load map data for rendering
        self._load_map_data()

        # Wrap with PufferLib for better multi-agent handling if needed
        if hasattr(self.env, "agents"):
            try:
                self._puffer_env = PettingZooPufferEnv(self.env)
            except Exception as e:
                logger.warning(f"Could not create PufferLib wrapper: {e}")
                self._puffer_env = None

        # Will be initialized on first reset
        self._controlled_agent = -1

    def _extract_agent_position(self, obs_agent: Dict[str, Any]) -> Tuple[int, int]:
        """Extract agent position from observation."""
        if "Entity" in obs_agent:
            entity_data = obs_agent["Entity"]
            if isinstance(entity_data, np.ndarray) and len(entity_data) > 0:
                # First entity should be the agent itself in NMMO
                if entity_data.shape[0] > 0:
                    # Assuming columns are [row, col, ...]
                    return (int(entity_data[0, 0]), int(entity_data[0, 1]))
        # Fallback position
        return (50, 50)

    def _extract_agent_facing(self, obs_agent: Dict[str, Any]) -> int:
        """Extract agent facing direction."""
        # NMMO might not have explicit facing, use 0 as default
        return 0

    def _extract_agent_health(self, obs_agent: Dict[str, Any]) -> int:
        """Extract agent health from observation."""
        if "Entity" in obs_agent:
            entity_data = obs_agent["Entity"]
            if isinstance(entity_data, np.ndarray) and len(entity_data) > 0:
                if entity_data.shape[0] > 0 and entity_data.shape[1] > 2:
                    # Health might be in column 2 or 3
                    return max(1, int(entity_data[0, 2]))  # Ensure at least 1 health
        # Fallback health
        return 100

    def _extract_agent_stamina(self, obs_agent: Dict[str, Any]) -> int:
        """Extract agent stamina from observation."""
        if "Entity" in obs_agent:
            entity_data = obs_agent["Entity"]
            if isinstance(entity_data, np.ndarray) and len(entity_data) > 0:
                if entity_data.shape[0] > 0 and entity_data.shape[1] > 3:
                    # Stamina might be in column 3 or 4
                    return max(1, int(entity_data[0, 3]))  # Ensure at least 1 stamina
        # Fallback stamina
        return 100

    # ------------------------------------------------------------------
    # Map loading and rendering utilities
    # ------------------------------------------------------------------

    def _load_map_data(self) -> None:
        """Load map data for rendering."""
        try:
            # Get the directory where this engine.py file is located
            engine_dir = Path(__file__).parent
            map_file = engine_dir / "maps" / "medium" / "map1" / "map.npy"

            if map_file.exists():
                self._map_data = np.load(map_file)
                logger.info(
                    f"Loaded map data from {map_file}, shape: {self._map_data.shape}"
                )
            else:
                logger.warning(f"Map file not found at {map_file}")
                self._map_data = None
        except Exception as e:
            logger.warning(f"Could not load map data: {e}")
            self._map_data = None

    def _render_image(self, public_state: NeuralMMOPublicState) -> np.ndarray:
        """Render the current game state as an RGB image."""
        if self._map_data is None:
            # Return a simple placeholder image
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 0] = 128  # Gray placeholder
            return img

        # Create color mapping for terrain types
        terrain_colors = {
            0: (139, 69, 19),  # Brown for lava/impassable
            1: (65, 105, 225),  # Blue for water
            2: (34, 139, 34),  # Green for grass
            3: (144, 238, 144),  # Light green for forest
            4: (210, 180, 140),  # Tan for desert
            5: (169, 169, 169),  # Gray for stone
            6: (255, 255, 224),  # Light yellow for plains
        }

        # Create base terrain image
        h, w = self._map_data.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        for terrain_id, color in terrain_colors.items():
            mask = self._map_data == terrain_id
            rgb_image[mask] = color

        # Add agent position if available
        if public_state.position:
            row, col = public_state.position
            if 0 <= row < h and 0 <= col < w:
                # Draw agent as red dot (3x3 square)
                r_start, r_end = max(0, row - 1), min(h, row + 2)
                c_start, c_end = max(0, col - 1), min(w, col + 2)
                rgb_image[r_start:r_end, c_start:c_end] = (255, 0, 0)  # Red

        # Scale down if image is too large for practical use
        if h > 256 or w > 256:
            # Simple downsampling by factor of 2 or 4
            scale_factor = 4 if h > 512 else 2
            new_h, new_w = h // scale_factor, w // scale_factor
            scaled_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            for i in range(new_h):
                for j in range(new_w):
                    # Take average of scale_factor x scale_factor region
                    region = rgb_image[
                        i * scale_factor : (i + 1) * scale_factor,
                        j * scale_factor : (j + 1) * scale_factor,
                    ]
                    scaled_img[i, j] = np.mean(region, axis=(0, 1)).astype(np.uint8)
            return scaled_img

        return rgb_image

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _select_agent_id(self, obs_batch: Dict[int, Dict[str, Any]]) -> int:
        """Pick the first living agent id from obs batch."""
        if self._controlled_agent in obs_batch:
            return self._controlled_agent
        # pick deterministic id for reproducibility
        return sorted(obs_batch.keys())[0]

    # ------------------------------------------------------------------
    # Internal state builders
    # ------------------------------------------------------------------

    def _build_public_state(self, obs_agent: Dict[str, Any]) -> NeuralMMOPublicState:
        # Entity is numpy array
        entity_data = obs_agent["Entity"]

        # Extract inventory data from agent observation
        inventory = {}
        if "Inventory" in obs_agent:
            inv_data = obs_agent["Inventory"]
            # Handle inventory as numpy array or dict
            if isinstance(inv_data, np.ndarray):
                # Map inventory indices to item names (NMMO specific)
                item_names = ["food", "water", "gold", "weapon", "armor"]
                for i, item_name in enumerate(item_names):
                    if i < len(inv_data):
                        inventory[item_name] = int(inv_data[i])
            elif isinstance(inv_data, dict):
                inventory = {str(k): int(v) for k, v in inv_data.items()}

        # Fallback to dummy data if no inventory found
        if not inventory:
            inventory = {"food": 100, "water": 100, "gold": 0}

        return NeuralMMOPublicState(
            tick=int(obs_agent.get("CurrentTick", 0)),
            num_steps_taken=int(obs_agent.get("CurrentTick", 0)),  # synonym
            max_episode_steps=int(self.env.config.TICK_LIMIT),
            agent_id=self._controlled_agent,
            position=self._extract_agent_position(obs_agent),
            facing=self._extract_agent_facing(obs_agent),
            health=self._extract_agent_health(obs_agent),
            stamina=self._extract_agent_stamina(obs_agent),
            inventory=inventory,
            local_terrain=obs_agent.get("Tile", np.zeros((15, 15), dtype=np.int16)),
            visible_entities=entity_data,
            team_score=0.0,  # PLACEHOLDER: Dummy score
            personal_score=0.0,  # PLACEHOLDER: Dummy score
        )

    def _build_private_state(
        self,
        reward: float,
        info_batch: Dict[str, Any],
        done_for_agent: bool,
        obs_for_agent: Dict[str, Any],
    ) -> NeuralMMOPrivateState:
        # PLACEHOLDER: Use dummy skill data for now
        skill_keys = [
            "melee",
            "range",
            "mage",
            "fishing",
            "hunting",
            "prospecting",
            "carving",
            "alchemy",
        ]

        skills = {
            k: SkillSnapshot(lvl=1, xp=0)  # PLACEHOLDER: Dummy skill values
            for k in skill_keys
        }

        achievements = info_batch.get("tasks", {}).get(self._controlled_agent, {})

        agent_is_dead = info_batch.get("dead", {}).get(self._controlled_agent, False)
        terminated = bool(done_for_agent and agent_is_dead)
        truncated = bool(done_for_agent and not agent_is_dead)

        return NeuralMMOPrivateState(
            reward_last_step=reward,
            total_reward_episode=self._total_reward,
            terminated=terminated,
            truncated=truncated,
            env_rng_state_snapshot={},  # PLACEHOLDER: Dummy RNG state for now
            skills=skills,
            achievements_status=achievements,
            agent_internal_stats={
                "hp": 100,  # PLACEHOLDER: Dummy values
                "stam": 100,  # PLACEHOLDER
                "gold": 0,  # PLACEHOLDER
            },
        )

    # ------------------------------------------------------------------
    # Core StatefulEngine API
    # ------------------------------------------------------------------

    async def _reset_engine(
        self, *, seed: Optional[int] | None = None
    ) -> Tuple[NeuralMMOPrivateState, NeuralMMOPublicState]:
        result = self.env.reset(seed=seed)
        # Handle both tuple and dict returns from env.reset()
        if isinstance(result, tuple):
            obs_batch, info = result
        else:
            obs_batch = result
        self._controlled_agent = self._select_agent_id(obs_batch)
        obs_agent = obs_batch[self._controlled_agent]

        self._total_reward = 0.0
        pub = self._build_public_state(obs_agent)
        # For reset, info_batch is empty, and agent is not done.
        priv = self._build_private_state(
            reward=0.0, info_batch={}, done_for_agent=False, obs_for_agent=obs_agent
        )
        # Store for checkpoint
        self._last_public_state = pub
        self._last_private_state = priv
        return priv, pub

    async def _step_engine(
        self, action: Dict[str, Any]
    ) -> Tuple[NeuralMMOPrivateState, NeuralMMOPublicState]:
        """`action` must already be a legal NMMO 3 action-dict."""
        action_batch = {self._controlled_agent: action}
        obs_batch, reward_batch, done_batch, info_batch = self.env.step(action_batch)

        r = float(reward_batch.get(self._controlled_agent, 0.0))
        self._total_reward += r

        obs_agent = obs_batch[self._controlled_agent]
        # Determine if the controlled agent is "done" for terminated/truncated logic
        done_for_agent_step = done_batch.get(self._controlled_agent, False)

        pub = self._build_public_state(obs_agent)
        priv = self._build_private_state(
            reward=r,
            info_batch=info_batch,
            done_for_agent=done_for_agent_step,
            obs_for_agent=obs_agent,
        )
        # Store for checkpoint
        self._last_public_state = pub
        self._last_private_state = priv
        return priv, pub

    async def _render(
        self,
        private_state: NeuralMMOPrivateState,
        public_state: NeuralMMOPublicState,
        get_observation: Optional[GetObservationCallable] = None,
    ) -> str:  # type: ignore[override]
        obs_cb = get_observation or NeuralMMOObservationCallable(self)
        obs = await obs_cb.get_observation(public_state, private_state)
        # stringify minimal preview
        return f"tick {public_state.tick} – hp {public_state.health} – total_r {private_state.total_reward_episode:.2f}"

    # ------------------------------------------------------------------
    # Snapshotting
    # ------------------------------------------------------------------

    async def _serialize_engine(self) -> NeuralMMOEngineSnapshot:
        blob = pickle.dumps(self.env)
        return NeuralMMOEngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            pickle_blob=blob,
            total_reward=self._total_reward,
            controlled_agent_id=self._controlled_agent,
            rng_state=getattr(
                self.env, "np_random", np.random.default_rng()
            ).bit_generator.state
            if hasattr(self.env, "np_random")
            else {},
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: "NeuralMMOEngineSnapshot"
    ) -> "NeuralMMOEngine":
        task_instance = await TaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls.__new__(cls)
        StatefulEngine.__init__(engine)
        engine.task_instance = task_instance
        engine.env = pickle.loads(snapshot.pickle_blob)
        engine._total_reward = snapshot.total_reward
        engine._controlled_agent = snapshot.controlled_agent_id
        # Restore NumPy RNG state for exact reproducibility if available
        if hasattr(engine.env, "np_random") and snapshot.rng_state:
            engine.env.np_random.bit_generator.state = snapshot.rng_state
        return engine
