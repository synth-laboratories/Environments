from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.tasks.core import TaskInstance


@dataclass
class MiniGridPublicState:
    mission: str
    image: np.ndarray
    agent_pos: Tuple[int, int]
    agent_dir: int
    step_count: int
    max_steps: int


@dataclass
class MiniGridPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool


@dataclass
class MiniGridEngineSnapshot(StatefulEngineSnapshot):
    env_state: dict


class MiniGridEngine(StatefulEngine):
    """Thin wrapper around gymnasium-minigrid environments."""

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        env_id = getattr(task_instance, "env_id", "MiniGrid-Empty-8x8-v0")
        self.env = gym.make(env_id)
        self._total_reward = 0.0

    def _build_public_state(self, obs: dict) -> MiniGridPublicState:
        mission = obs.get("mission", "")
        image = obs.get("image")
        agent_pos = tuple(int(x) for x in self.env.agent_pos)
        agent_dir = int(self.env.agent_dir)
        step_count = int(getattr(self.env, "step_count", 0))
        max_steps = int(getattr(self.env, "max_steps", 0))
        return MiniGridPublicState(
            mission=mission,
            image=image,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            step_count=step_count,
            max_steps=max_steps,
        )

    async def _reset_engine(self) -> tuple[MiniGridPrivateState, MiniGridPublicState]:
        obs, _ = self.env.reset()
        self._total_reward = 0.0
        priv = MiniGridPrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
        )
        pub = self._build_public_state(obs)
        return priv, pub

    async def _step_engine(
        self, action: int
    ) -> tuple[MiniGridPrivateState, MiniGridPublicState]:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self._total_reward += float(reward)
        priv = MiniGridPrivateState(
            reward_last=float(reward),
            total_reward=self._total_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        pub = self._build_public_state(obs)
        return priv, pub

    async def _serialize_engine(self) -> MiniGridEngineSnapshot:
        state = self.env.unwrapped.state
        return MiniGridEngineSnapshot(env_state=state)

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: MiniGridEngineSnapshot, task_instance: TaskInstance
    ) -> "MiniGridEngine":
        eng = cls(task_instance)
        eng.env.reset()
        eng.env.unwrapped.state = snapshot.env_state
        return eng
