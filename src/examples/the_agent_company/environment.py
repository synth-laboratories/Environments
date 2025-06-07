from typing import List
from src.stateful.core import StatefulEnvironment
from src.environment.tools import EnvToolCall

from .taskset import TACTaskInstance


class TACEnvironment(StatefulEnvironment):
    """Minimal stub environment for The Agent Company tasks."""

    def __init__(self, task_instance: TACTaskInstance):
        self.task_instance = task_instance
        self._step = 0

    def validate_tool_calls(self, tool_calls: EnvToolCall):
        pass

    async def initialize(self):
        self._step = 0
        return {"task_image": self.task_instance.metadata.image, "step": self._step}

    async def step(self, tool_calls: List[EnvToolCall]):
        self._step += 1
        return {"step": self._step}

    async def checkpoint(self):
        return {"step": self._step}

    async def terminate(self):
        return {"terminated": True, "step": self._step}
