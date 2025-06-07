from __future__ import annotations

import numpy as np
import bsuite
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.tasks.core import TaskInstance


@dataclass
class BSuiteEngineSnapshot(StatefulEngineSnapshot):
    bsuite_id: str
    total_reward: float


@dataclass
class BSuitePublicState:
    observation: Any
    step_type: int
    reward: float
    discount: float


@dataclass
class BSuitePrivateState:
    total_reward: float
    terminated: bool


class BSuiteEngine(StatefulEngine):
    def __init__(self, task_instance: TaskInstance) -> None:
        self.task_instance = task_instance
        self.bsuite_id = getattr(task_instance, "bsuite_id", "catch/0")
        if not hasattr(np, "int"):
            np.int = int
        self.env = bsuite.load_from_id(self.bsuite_id)
        self.total_reward = 0.0
        self._last_timestep = None

    def _timestep_to_public(self, ts) -> BSuitePublicState:
        obs = ts.observation
        if hasattr(obs, "tolist"):
            obs = obs.tolist()
        return BSuitePublicState(
            observation=obs,
            step_type=int(getattr(ts, "step_type", 1)),
            reward=float(getattr(ts, "reward", 0.0)),
            discount=float(getattr(ts, "discount", 1.0)),
        )

    async def _reset_engine(
        self, *, seed: Optional[int] = None
    ) -> Tuple[BSuitePrivateState, BSuitePublicState]:
        self.total_reward = 0.0
        self._last_timestep = self.env.reset()
        priv = BSuitePrivateState(total_reward=0.0, terminated=False)
        pub = self._timestep_to_public(self._last_timestep)
        return priv, pub

    async def _step_engine(self, action: int) -> Tuple[BSuitePrivateState, BSuitePublicState]:
        self._last_timestep = self.env.step(action)
        self.total_reward += float(getattr(self._last_timestep, "reward", 0.0))
        terminated = int(getattr(self._last_timestep, "step_type", 1)) == 2
        priv = BSuitePrivateState(total_reward=self.total_reward, terminated=terminated)
        pub = self._timestep_to_public(self._last_timestep)
        return priv, pub

    async def _serialize_engine(self) -> BSuiteEngineSnapshot:
        return BSuiteEngineSnapshot(bsuite_id=self.bsuite_id, total_reward=self.total_reward)

    @classmethod
    async def _deserialize_engine(cls, snapshot: BSuiteEngineSnapshot, task_instance: TaskInstance) -> "BSuiteEngine":
        task_instance.bsuite_id = snapshot.bsuite_id
        eng = cls(task_instance)
        eng.total_reward = snapshot.total_reward
        return eng
