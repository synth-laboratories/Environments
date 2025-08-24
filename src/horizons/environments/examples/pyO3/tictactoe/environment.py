from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....stateful.core import StatefulEnvironment
from ....environment.shared_engine import InternalObservation
from ....environment.tools import EnvToolCall


def _to_internal(obs_dict: Dict[str, Any]) -> InternalObservation:
    """Convert PyO3-returned dict to InternalObservation.

    Supports two shapes:
    - New (preferred): {"public_observation": {...}, "private_observation": {...}}
    - Legacy: {"terminated": bool, "truncated": bool, "data": {...}}
    """
    obs = InternalObservation()
    if "public_observation" in obs_dict:
        obs.public_observation = dict(obs_dict.get("public_observation", {}))
        obs.private_observation = dict(obs_dict.get("private_observation", {}))
        return obs
    # Legacy fallback
    data = dict(obs_dict.get("data", {}))
    data.setdefault("terminated", bool(obs_dict.get("terminated", False)))
    data.setdefault("truncated", bool(obs_dict.get("truncated", False)))
    obs.public_observation = data
    obs.private_observation = {}
    return obs


class RustTicTacToeEnvironment(StatefulEnvironment):
    """PyO3-backed TicTacToe environment (Rust engine)."""

    def __init__(self, task_instance: Any, config: Optional[Dict[str, Any]] = None):
        from horizons_env_py import create_rust_env  # type: ignore

        cfg = config or {}
        init = getattr(task_instance, "initial_engine_snapshot", {}) or {}

        # Build config payload for Rust env
        rust_cfg = {
            "agent_mark": init.get("agent_mark") or cfg.get("agent_mark") or "X",
            "opponent_minimax_prob": init.get("opponent_minimax_prob") or cfg.get("opponent_minimax_prob") or 0.0,
            "seed": init.get("seed") or cfg.get("seed") or 42,
        }

        # Create generic rust env instance by name
        self._env = create_rust_env("TicTacToe", rust_cfg)

    async def initialize(self) -> InternalObservation:
        raw = self._env.initialize()
        return _to_internal(raw)

    async def terminate(self) -> InternalObservation:
        raw = self._env.terminate()
        return _to_internal(raw)

    def validate_tool_calls(self, tool_calls: List[EnvToolCall]):
        for tc in tool_calls:
            if tc.tool != "interact":
                raise ValueError(f"Unknown tool: {tc.tool}")
            letter = (tc.args or {}).get("letter")
            number = (tc.args or {}).get("number")
            if not letter or number is None:
                raise ValueError("Both 'letter' and 'number' parameters are required for interact tool")
            if str(letter).upper() not in ["A", "B", "C"]:
                raise ValueError(f"Invalid letter '{letter}'. Must be A, B, or C")
            if number not in [1, 2, 3]:
                raise ValueError(f"Invalid number '{number}'. Must be 1, 2, or 3")

    async def step(self, tool_calls: List[EnvToolCall]) -> InternalObservation:
        self.validate_tool_calls(tool_calls)
        tc = tool_calls[0]
        raw = self._env.step_tool("interact", {"letter": tc.args["letter"], "number": tc.args["number"]})  # type: ignore
        return _to_internal(raw)

    async def checkpoint(self) -> InternalObservation:
        snap = self._env.checkpoint()
        obs = InternalObservation()
        obs.public_observation = {"snapshot": snap}
        obs.private_observation = {}
        return obs
