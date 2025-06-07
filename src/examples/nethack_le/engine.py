from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from nle.env.base import NLE

from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.reproducibility.core import IReproducibleEngine
from src.tasks.core import TaskInstance


@dataclass
class NethackEngineSnapshot(StatefulEngineSnapshot):
    seeds: Tuple[int, int, bool, Optional[int]]
    total_reward: float
    steps: int


@dataclass
class NethackPublicState:
    tty_chars: np.ndarray
    tty_colors: np.ndarray
    tty_cursor: np.ndarray
    message: str
    glyphs: Optional[np.ndarray] = None
    step: int = 0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NethackPrivateState:
    reward_last_step: float
    total_reward: float
    terminated: bool
    truncated: bool


class NethackEngine(StatefulEngine, IReproducibleEngine):
    """Thin wrapper around nle.env.base.NLE providing step/reset helpers."""

    def __init__(self, task_instance: TaskInstance, render_mode: str = "ansi"):
        self.task_instance = task_instance
        self.render_mode = render_mode
        self.env = NLE(render_mode=render_mode)
        self.total_reward = 0.0
        self.steps = 0
        self.last_observation: Optional[Dict[str, Any]] = None

    async def _step_engine(
        self, action: int
    ) -> Tuple[NethackPrivateState, NethackPublicState]:
        obs, reward, done, truncated, info = self.env.step(action)
        self.last_observation = obs
        self.total_reward += float(reward)
        self.steps += 1

        message = bytes(obs["message"]).split(b"\0", 1)[0].decode("utf-8")

        priv = NethackPrivateState(
            reward_last_step=float(reward),
            total_reward=self.total_reward,
            terminated=bool(done),
            truncated=bool(truncated),
        )
        pub = NethackPublicState(
            tty_chars=obs["tty_chars"],
            tty_colors=obs["tty_colors"],
            tty_cursor=obs["tty_cursor"],
            message=message,
            glyphs=obs.get("glyphs"),
            step=self.steps,
            info=info,
        )
        return priv, pub

    async def _reset_engine(
        self, seed: int | None = None
    ) -> Tuple[NethackPrivateState, NethackPublicState]:
        obs, info = self.env.reset(seed=seed)
        self.total_reward = 0.0
        self.steps = 0
        self.last_observation = obs

        message = bytes(obs["message"]).split(b"\0", 1)[0].decode("utf-8")

        priv = NethackPrivateState(
            reward_last_step=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
        )
        pub = NethackPublicState(
            tty_chars=obs["tty_chars"],
            tty_colors=obs["tty_colors"],
            tty_cursor=obs["tty_cursor"],
            message=message,
            glyphs=obs.get("glyphs"),
            step=0,
            info=info,
        )
        return priv, pub

    async def _serialize_engine(self) -> NethackEngineSnapshot:
        seeds = self.env.get_seeds()
        return NethackEngineSnapshot(
            seeds=seeds, total_reward=self.total_reward, steps=self.steps
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: NethackEngineSnapshot, task_instance: TaskInstance
    ) -> "NethackEngine":
        eng = cls(task_instance)
        eng.env.seed(*snapshot.seeds)
        obs, _ = eng.env.reset()
        eng.total_reward = snapshot.total_reward
        eng.steps = snapshot.steps
        eng.last_observation = obs
        return eng
