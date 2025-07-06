"""
AlgoTune integration for the synth-env framework.

Requires:
    pip install -e git+https://github.com/oripress/AlgoTune.git#egg=algotune
"""

from __future__ import annotations

import importlib
import inspect
import textwrap
import time
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from synth_env.environment.registry import register_environment
from synth_env.environment.shared_engine import (
    InternalObservation,
    GetObservationCallable,
)
from synth_env.environment.tools import AbstractTool, EnvToolCall, ToolResult
from synth_env.reproducibility.core import ReproducibleEnvironment
from synth_env.stateful.core import StatefulEnvironment
from synth_env.tasks.core import TaskInstance

# ------------------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------------------


class _AlgoTuneEngine:
    """Minimal "engine" wrapper around a single AlgoTune task instance."""

    def __init__(
        self,
        task_name: str,
        n: int = 128,
        random_seed: int = 1,
    ):
        self.task_name = task_name
        self.n = n
        self.random_seed = random_seed
        # Import TaskFactory from AlgoTune
        from AlgoTuneTasks.factory import TaskFactory

        # Create the task instance
        self._task = TaskFactory(task_name)

        # Generate problem and solve for baseline
        self.problem = self._task.generate_problem(n, random_seed=random_seed)
        t0 = time.perf_counter()
        self.baseline = self._task.solve(self.problem)
        self.baseline_time = time.perf_counter() - t0
        self.best_solution: Optional[Any] = None
        self.best_time: float = float("inf")
        self.attempts: int = 0

    # ------------------------------------------------------------------ helpers

    def _evaluate_candidate(self, solve_fn) -> Tuple[bool, float]:
        """Run candidate solver, return (is_correct, elapsed_seconds)."""
        t0 = time.perf_counter()
        try:
            candidate_sol = solve_fn(self.problem)
        except Exception:
            return False, float("inf")
        elapsed = time.perf_counter() - t0
        is_ok = self._task.is_solution(self.problem, candidate_sol)
        return is_ok, elapsed

    # ----------------------------------------------------------- external API

    async def reset(self):
        """Return initial public / private state."""
        return (
            {"terminated": False},
            {
                "task": self.task_name,
                "n": self.n,
                "baseline_time": self.baseline_time,
                "attempts": 0,
            },
        )

    async def step(self, code: str):
        """Run a code candidate, update state, and return new observation."""
        self.attempts += 1
        scope: Dict[str, Any] = {}
        exec(code, scope)  # noqa: S102  (trusted researcher use)
        if "solve" not in scope or not inspect.isfunction(scope["solve"]):
            return False, "No `solve` function defined", {}
        ok, t = self._evaluate_candidate(scope["solve"])
        if ok and t < self.best_time:
            self.best_time = t
            self.best_solution = scope["solve"]
        speedup = self.baseline_time / t if ok else 0.0
        pub = {
            "attempt": self.attempts,
            "ok": ok,
            "elapsed": t,
            "speedup_vs_baseline": speedup,
            "best_speedup": self.baseline_time / self.best_time
            if self.best_time < float("inf")
            else 0.0,
        }
        priv = {"terminated": False}
        return ok, None, (priv, pub)

    async def terminate(self):
        return ({"terminated": True}, {"terminated": True})


# ------------------------------------------------------------------------------
# Tooling layer
# ------------------------------------------------------------------------------


class AlgoTuneCodeInput(BaseModel):
    """Arguments schema for the optimise tool."""

    code: str = Field(
        description="A self-contained Python snippet that defines a function `solve(problem)`."
    )


class _AlgoTuneTool(AbstractTool):
    name = "optimise"
    description = (
        "Submit candidate `solve(problem)` implementation for speed optimisation."
    )
    call_schema = AlgoTuneCodeInput
    result_schema = ToolResult

    def __init__(self, engine: _AlgoTuneEngine):
        self._engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            code = call.args["code"]
            ok, err, (priv, pub) = await self._engine.step(code)
            return ToolResult(ok=ok, error=err, payload={"public_state": pub})
        except Exception as e:  # pragma: no cover
            return ToolResult(ok=False, error=str(e), payload={})


# ------------------------------------------------------------------------------
# Environment wrapper
# ------------------------------------------------------------------------------


class AlgoTuneEnvironment(
    StatefulEnvironment, ReproducibleEnvironment[_AlgoTuneEngine]
):
    """Synth-env wrapper exposing AlgoTune as a one-shot optimisation task."""

    def __init__(
        self,
        task_instance: TaskInstance,
        custom_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "AlgoTune"
        self.task_instance = task_instance

        # Extract parameters from task instance metadata
        from synth_env.examples.algotune.taskset import AlgoTuneTaskInstanceMetadata

        if isinstance(task_instance.metadata, AlgoTuneTaskInstanceMetadata):
            task_name = task_instance.metadata.task_name
            n = task_instance.metadata.problem_size
            random_seed = task_instance.metadata.random_seed
        else:
            # Fallback for compatibility
            task_name = getattr(
                task_instance.metadata, "task_name", "matrix_multiplication"
            )
            n = getattr(task_instance.metadata, "problem_size", 128)
            random_seed = getattr(task_instance.metadata, "random_seed", 1)

        self.engine = _AlgoTuneEngine(task_name, n, random_seed)
        self._optimise_tool = _AlgoTuneTool(self.engine)
        self._obs_callable = custom_obs

    # --------------------------------------------------- stateful interface

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine.reset()
        return {"public": pub, "private": priv}

    async def step(self, tool_calls):  # type: ignore[override]
        call = self.validate_tool_calls(tool_calls)
        result = await self._optimise_tool(call)
        priv = {}
        pub = result.payload.get("public_state", {})
        if not result.ok:
            pub["error"] = result.error
        return {"public": pub, "private": priv}

    async def checkpoint(self) -> InternalObservation:
        # Snapshot = current best stats
        return await self.initialize()

    async def terminate(self) -> InternalObservation:
        priv, pub = await self.engine.terminate()
        return {"public": pub, "private": priv}

    # --------------------------------------------------- helpers

    def validate_tool_calls(self, tool_calls) -> EnvToolCall:  # type: ignore[override]
        if isinstance(tool_calls, EnvToolCall):
            return tool_calls
        if isinstance(tool_calls, str):
            raise ValueError("Provide a dict with a `code` field.")
        if isinstance(tool_calls, dict) and "code" in tool_calls:
            return EnvToolCall(tool="optimise", args={"code": tool_calls["code"]})
        raise ValueError("Unrecognised tool call format.")

    # ------------------------------------------------ reproduction hooks

    async def _serialize_engine(self) -> Dict[str, Any]:
        return {
            "task": self.engine.task_name,
            "n": self.engine.n,
            "seed": self.engine.random_seed,
        }

    @classmethod
    async def _deserialize_engine(cls, task_instance, snapshot):
        return _AlgoTuneEngine(
            snapshot["task"], snapshot["n"], random_seed=snapshot["seed"]
        )


# Register in global registry
register_environment("AlgoTune", AlgoTuneEnvironment)
