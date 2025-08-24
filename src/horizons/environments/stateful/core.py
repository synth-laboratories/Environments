from abc import abstractmethod
from typing import List, Any, Tuple

from ...environment.shared_engine import Engine, InternalObservation
from ...environments.environment.tools import EnvToolCall


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
        """Current step method - takes tool calls and returns InternalObservation."""
        pass

    def step_gymnasium(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """
        Gymnasium-compliant step method.

        Args:
            action: Single action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to tool call format
        tool_call = EnvToolCall(
            tool="interact",  # Default tool for action-based environments
            args={"action": action}
        )

        # Use asyncio to run the async step method synchronously
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async step method
        result = loop.run_until_complete(self.step([tool_call]))

        # Extract gymnasium format from result
        observation = result.public_observation if hasattr(result, 'public_observation') else result
        reward = observation.get("reward_last", 0.0) if isinstance(observation, dict) else 0.0
        terminated = observation.get("terminated", False) if isinstance(observation, dict) else False
        truncated = observation.get("truncated", False) if isinstance(observation, dict) else False

        info = {
            "terminated": terminated,
            "truncated": truncated,
        }

        return observation, reward, terminated, truncated, info

    @abstractmethod
    async def checkpoint(self) -> InternalObservation:
        pass
