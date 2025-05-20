"""examples/nmmo3/engine.py
NMMO3Engine — stateful, reproducible wrapper around pufferlib+Neural-MMO 3
----------------------------------------------------------------------------
A drop-in sibling to CrafterEngine / SokobanEngine.  The design goals are:
  • *Single-agent control* – we control exactly one agent (``agent_id``)
  • *Bit-exact replay*    – full snapshot via ``pickle`` of the underlying
    ``pufferlib.emulation.PufferEnv`` instance + numpy RNG state.
  • *Minimal surface*     – expose only what the agent needs (see PublicState)
    while maintaining enough PrivateState to validate determinism.

If you need a multi-agent wrapper, keep the same structure but replace all
``agent_id`` scalars with lists and iterate over the per-agent observation
dicts returned by the environment.
"""

from __future__ import annotations

import pickle, logging, copy
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import nmmo  # pip install pufferlib[nmmo]
import pufferlib.emulation as pe
from pufferlib.emulation import PufferEnv

from stateful.engine import StatefulEngine, StatefulEngineSnapshot
from reproducibility.core import IReproducibleEngine
from environment.shared_engine import GetObservationCallable, InternalObservation
from tasks.core import TaskInstance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ════════════════════════════════════════════════════════════════════════════
#  Snapshot + runtime state dataclasses
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class NMMO3EngineSnapshot(StatefulEngineSnapshot):
    """Binary snapshot – stores the *entire* PufferEnv via pickle plus engine state."""

    task_instance_dict: Dict[str, Any]
    pickle_blob: bytes
    total_reward: float
    controlled_agent_id: int
    rng_state: Dict[str, Any]  # NumPy RNG state for reproducibility


@dataclass
class NMMO3PublicState:
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
    def diff(self, prev: "NMMO3PublicState") -> Dict[str, Any]:
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
    lvl:  int   # current level
    xp:   int   # accumulated XP


@dataclass
class NMMO3PrivateState:
    # ─── rewards / termination ──────────────────────────────────────────────
    reward_last_step:      float
    total_reward_episode:  float
    terminated:            bool
    truncated:             bool

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
    def diff(self, prev: "NMMO3PrivateState") -> Dict[str, Any]:
        changed: Dict[str, Any] = {}
        for f in self.__dataclass_fields__:
            new, old = getattr(self, f), getattr(prev, f)
            if new != old:
                changed[f] = (old, new)
        return changed


# ════════════════════════════════════════════════════════════════════════════
#  Observation helper
# ════════════════════════════════════════════════════════════════════════════


class NMMO3ObservationCallable(GetObservationCallable):
    """Default observation transformer → returns a plain dict."""

    async def get_observation(  # type: ignore[override]
        self, pub: NMMO3PublicState, priv: NMMO3PrivateState
    ) -> InternalObservation:
        return {
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


# ════════════════════════════════════════════════════════════════════════════
#  Engine implementation
# ════════════════════════════════════════════════════════════════════════════


class NMMO3Engine(StatefulEngine, IReproducibleEngine):
    """Wraps a *single-agent* PufferEnv instance in the Engine API."""

    task_instance: TaskInstance
    env: PufferEnv
    _total_reward: float
    _controlled_agent: int  # the id we act for
    # Last known states for checkpointing
    _last_public_state: NMMO3PublicState | None
    _last_private_state: NMMO3PrivateState | None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self._total_reward = 0.0
        # Initialize last states
        self._last_public_state = None
        self._last_private_state = None

        # --- build env ----------------------------------------------------
        cfg_dict = copy.deepcopy(getattr(task_instance, "config", {}) or {})
        # Reasonable defaults ensuring reproducibility
        cfg_dict.setdefault("num_envs", 1)
        cfg_dict.setdefault("tick_limit", 1_000)
        cfg_dict.setdefault("map_size", 128)
        cfg_dict.setdefault("seed", None)

        self.env: PufferEnv = pe.make("nmmo", **cfg_dict)

        # Will be initialised on first reset
        self._controlled_agent = -1

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

    def _build_public_state(self, obs_agent: Dict[str, Any]) -> NMMO3PublicState:
        ent_row = obs_agent["entities"]  # pandas.DataFrame row (single agent)
        # convert to plain dict / ints for safety
        inventory = {
            "food": int(ent_row.food),
            "water": int(ent_row.water),
            "gold": int(ent_row.gold),
        }

        return NMMO3PublicState(
            tick=int(obs_agent["tick"]),
            num_steps_taken=int(obs_agent["tick"]),  # synonym
            max_episode_steps=int(self.env.unwrapped.config.TICK_LIMIT),
            agent_id=self._controlled_agent,
            position=(int(ent_row.row), int(ent_row.col)),
            facing=int(ent_row.orientation),
            health=int(ent_row.health),
            stamina=int(ent_row.stamina),
            inventory=inventory,
            local_terrain=np.asarray(obs_agent["terrain"], dtype=np.int16),
            visible_entities=np.asarray(obs_agent["entities"].to_numpy()),
            team_score=float(obs_agent["stats"].get("team_score", 0.0)),
            personal_score=float(obs_agent["stats"].get("score", 0.0)),
        )

    def _build_private_state(
        self,
        reward: float,
        info_batch: Dict[str, Any],
        done_for_agent: bool,
        obs_for_agent: Dict[str, Any],
    ) -> NMMO3PrivateState:
        ent_series = obs_for_agent["entities"]
        skill_keys = [
            "melee", "range", "mage", "fishing", "hunting",
            "prospecting", "carving", "alchemy",
        ]

        skills = {
            k: SkillSnapshot(
                lvl=int(getattr(ent_series, f"skill_{k}_lvl", 0)),
                xp=int(getattr(ent_series, f"skill_{k}_xp", 0)),
            )
            for k in skill_keys
        }

        achievements = info_batch.get("tasks", {}).get(self._controlled_agent, {})
        # If tasks are not per-agent in info_batch['tasks'], use:
        # achievements = info_batch.get("tasks", {})

        agent_is_dead = info_batch.get("dead", {}).get(self._controlled_agent, False)
        terminated = bool(done_for_agent and agent_is_dead)
        truncated = bool(done_for_agent and not agent_is_dead)

        return NMMO3PrivateState(
            reward_last_step=reward,
            total_reward_episode=self._total_reward,
            terminated=terminated,
            truncated=truncated,
            env_rng_state_snapshot=self.env.np_random.bit_generator.state,
            skills=skills,
            achievements_status=achievements,
            agent_internal_stats={
                "hp": int(ent_series.health),
                "stam": int(ent_series.stamina),
                "gold": int(ent_series.gold),
            },
        )

    # ------------------------------------------------------------------
    # Core StatefulEngine API
    # ------------------------------------------------------------------

    async def _reset_engine(
        self, *, seed: Optional[int] | None = None
    ) -> Tuple[NMMO3PrivateState, NMMO3PublicState]:
        obs_batch = self.env.reset(seed=seed)
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
    ) -> Tuple[NMMO3PrivateState, NMMO3PublicState]:
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
        private_state: NMMO3PrivateState,
        public_state: NMMO3PublicState,
        get_observation: Optional[GetObservationCallable] = None,
    ) -> str:  # type: ignore[override]
        obs_cb = get_observation or NMMO3ObservationCallable()
        obs = await obs_cb.get_observation(public_state, private_state)
        # stringify minimal preview
        return f"tick {public_state.tick} – hp {public_state.health} – total_r {private_state.total_reward_episode:.2f}"

    # ------------------------------------------------------------------
    # Snapshotting
    # ------------------------------------------------------------------

    async def _serialize_engine(self) -> NMMO3EngineSnapshot:
        blob = pickle.dumps(self.env)
        return NMMO3EngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            pickle_blob=blob,
            total_reward=self._total_reward,
            controlled_agent_id=self._controlled_agent,
            rng_state=self.env.np_random.bit_generator.state,
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: "NMMO3EngineSnapshot"
    ) -> "NMMO3Engine":
        task_instance = await TaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls.__new__(cls)
        StatefulEngine.__init__(engine)
        engine.task_instance = task_instance
        engine.env = pickle.loads(snapshot.pickle_blob)
        engine._total_reward = snapshot.total_reward
        engine._controlled_agent = snapshot.controlled_agent_id
        # Restore NumPy RNG state for exact reproducibility
        engine.env.np_random.bit_generator.state = snapshot.rng_state
        return engine
