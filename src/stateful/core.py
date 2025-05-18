from src.environment.core import Environment
from src.environment.shared_engine import InternalObservation
from src.stateful.state import State
from typing import List
from src.environment.tools import EnvToolCall
from abc import abstractmethod


class StatefulEnvironment(Environment):
    @abstractmethod
    async def initialize(self) -> InternalObservation:
        pass

    @abstractmethod
    async def terminate(self) -> InternalObservation:
        pass

    # main external api
    @abstractmethod
    def validate_tool_calls(self, EnvToolCall):
        pass

    @abstractmethod
    async def step(self, tool_calls: List[List[EnvToolCall]]) -> InternalObservation:
        pass

    @abstractmethod
    async def checkpoint(self) -> InternalObservation:
        pass
