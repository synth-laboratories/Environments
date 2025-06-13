from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from src.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from src.tasks.core import TaskInstance
from src.reproducibility.core import IReproducibleEngine


@dataclass
class TreasureHunterPublicState:
    position: Tuple[int, int]
    has_treasure: bool
    step_count: int
    max_steps: int
    error_info: Optional[str] = None


@dataclass
class TreasureHunterPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool


@dataclass
class TreasureHunterEngineSnapshot(StatefulEngineSnapshot):
    state: Dict[str, Any]


class TreasureHunterEngine(StatefulEngine, IReproducibleEngine):
    """Simple text world engine where the agent must find a treasure."""

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self.grid_size = getattr(task_instance.metadata, "grid_size", 5)
        self.max_steps = getattr(task_instance.metadata, "max_steps", 20)
        self.treasure_pos = getattr(task_instance.metadata, "treasure_pos", (self.grid_size - 1, self.grid_size - 1))
        self._total_reward = 0.0
        self._step_count = 0
        self.position: Tuple[int, int] = (0, 0)
        self.has_treasure = False
        self.terminated = False
        self.truncated = False

    def _build_public_state(self) -> TreasureHunterPublicState:
        return TreasureHunterPublicState(
            position=self.position,
            has_treasure=self.has_treasure,
            step_count=self._step_count,
            max_steps=self.max_steps,
        )

    def _build_private_state(self, reward: float) -> TreasureHunterPrivateState:
        return TreasureHunterPrivateState(
            reward_last=reward,
            total_reward=self._total_reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    async def _reset_engine(self) -> tuple[TreasureHunterPrivateState, TreasureHunterPublicState]:
        self._total_reward = 0.0
        self._step_count = 0
        self.position = (0, 0)
        self.has_treasure = False
        self.terminated = False
        self.truncated = False
        priv = self._build_private_state(0.0)
        pub = self._build_public_state()
        return priv, pub

    async def _step_engine(self, command: str) -> tuple[TreasureHunterPrivateState, TreasureHunterPublicState]:
        if self.terminated:
            return self._build_private_state(0.0), self._build_public_state()

        reward = 0.0
        cmd = command.lower().strip()
        x, y = self.position
        if cmd == "north" and y > 0:
            y -= 1
        elif cmd == "south" and y < self.grid_size - 1:
            y += 1
        elif cmd == "west" and x > 0:
            x -= 1
        elif cmd == "east" and x < self.grid_size - 1:
            x += 1
        elif cmd == "take" and (x, y) == self.treasure_pos and not self.has_treasure:
            self.has_treasure = True
            reward = 1.0
            self.terminated = True
        else:
            # invalid command or no effect
            pass

        self.position = (x, y)
        self._step_count += 1
        self._total_reward += reward
        if self._step_count >= self.max_steps and not self.terminated:
            self.terminated = True
            self.truncated = True

        return self._build_private_state(reward), self._build_public_state()

    async def _serialize_engine(self) -> TreasureHunterEngineSnapshot:
        state = {
            "position": self.position,
            "has_treasure": self.has_treasure,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        return TreasureHunterEngineSnapshot(state=state)

    @classmethod
    async def _deserialize_engine(cls, snapshot: TreasureHunterEngineSnapshot, task_instance: TaskInstance) -> "TreasureHunterEngine":
        eng = cls(task_instance)
        state = snapshot.state
        eng.position = tuple(state.get("position", (0, 0)))
        eng.has_treasure = state.get("has_treasure", False)
        eng._step_count = state.get("step_count", 0)
        eng._total_reward = state.get("total_reward", 0.0)
        eng.terminated = state.get("terminated", False)
        eng.truncated = state.get("truncated", False)
        return eng
