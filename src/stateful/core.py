from abc import abstractmethod
from typing import List

from environment.shared_engine import Engine, InternalObservation
from environment.tools import EnvToolCall
from .state import State


class StatefulEnvironment(Engine):
    @abstractmethod
    async def initialize(self) -> InternalObservation:
        pass

    @abstractmethod
    async def terminate(self) -> InternalObservation:
        pass

    # main external api
    @abstractmethod
    def validate_tool_calls(self, tool_calls: EnvToolCall):
        pass

    @abstractmethod
    async def step(self, tool_calls: List[EnvToolCall]) -> InternalObservation:
        pass

    @abstractmethod
    async def checkpoint(self) -> InternalObservation:
        pass
