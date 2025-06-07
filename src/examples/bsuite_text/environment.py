from __future__ import annotations

from typing import Any

import numpy as np
import bsuite

from src.stateful.core import StatefulEnvironment
from src.environment.shared_engine import InternalObservation


class BSuiteTextEnvironment(StatefulEnvironment):
    """A lightweight wrapper around bsuite environments using a text interface."""

    def __init__(self, bsuite_id: str) -> None:
        self.name = "BSuiteText"
        self.bsuite_id = bsuite_id
        if not hasattr(np, "int"):
            np.int = int  # bsuite expects the deprecated np.int
        self.env = bsuite.load_from_id(bsuite_id)
        self._last_timestep = None

    async def initialize(self) -> InternalObservation:
        self._last_timestep = self.env.reset()
        return self._to_observation(self._last_timestep)

    async def terminate(self) -> InternalObservation:
        return {"terminated": True}

    def validate_tool_calls(self, tool_calls: Any) -> int:
        action: Any = None
        if isinstance(tool_calls, list):
            first = tool_calls[0]
            if isinstance(first, list):
                first = first[0]
            if isinstance(first, dict):
                action = first.get("action")
            else:
                action = first
        elif isinstance(tool_calls, dict):
            action = tool_calls.get("action")
        else:
            action = tool_calls
        if action is None:
            raise ValueError("Missing 'action' for bsuite step call")
        return int(action)

    async def step(self, tool_calls: Any) -> InternalObservation:
        action = self.validate_tool_calls(tool_calls)
        self._last_timestep = self.env.step(action)
        return self._to_observation(self._last_timestep)

    async def checkpoint(self) -> InternalObservation:
        if self._last_timestep is None:
            return {"error": "Environment not initialized"}
        return self._to_observation(self._last_timestep)

    def _to_observation(self, timestep) -> InternalObservation:
        obs = timestep.observation
        if hasattr(obs, "tolist"):
            obs = obs.tolist()
        return {
            "step_type": int(timestep.step_type),
            "reward": timestep.reward,
            "discount": timestep.discount,
            "observation": obs,
        }
