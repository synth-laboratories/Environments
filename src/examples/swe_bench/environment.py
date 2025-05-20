# examples/swe_bench/environment.py
"""
SWE-bench Environment
─────────────────────
Thin wrapper that exposes **SWEBenchEngine** through the generic
`StatefulEnvironment` interface so a ReAct-style agent can interact with
SWE-bench (or SWE-Gym) tasks via *function-calling* tools.

Available tools
---------------
* **read_file**    – read the contents (or a slice) of a file in the repo
* **apply_patch**  – apply a unified-diff patch to the workspace
* **run_tests**    – run (a subset of) the task's tests inside Docker
* **submit**       – convenience alias for `run_tests` followed by termination
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from environment.tools import EnvToolCall
from environment.shared_engine import GetObservationCallable, InternalObservation
from stateful.core import StatefulEnvironment
from examples.swe_bench.engine import SWEBenchEngine


# ─────────────────────────── Pydantic arg-schemas ────────────────────────────
class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Path (relative to repo root) of the file")
    start: int | None = Field(1, description="1-based first line to return (inclusive)")
    end: int | None = Field(400, description="1-based last line (inclusive)")


class ApplyPatchArgs(BaseModel):
    patch: str = Field(..., description="Unified diff patch (as a string)")
    reasoning: str | None = Field(
        None, description="(Optional) short LM justification for the patch"
    )


class RunTestsArgs(BaseModel):
    tests: Optional[List[str]] = Field(
        None,
        description="Optional list of pytest identifiers to run. "
        "Default = task's fail+pass tests.",
    )


class SubmitArgs(BaseModel):
    reason: str = Field(..., description="Why the agent believes the bug is fixed")


# ─────────────────────────── EnvToolCall wrappers ────────────────────────────
class ReadFile(EnvToolCall):
    """read_file(path,start?,end?)"""

    def __init__(self, **kwargs):  # kwargs come from ReadFileArgs
        args = ReadFileArgs(**kwargs).model_dump()
        self.action: Dict[str, Any] = {"type": "read_file", **args}


class ApplyPatch(EnvToolCall):
    """apply_patch(patch,…)"""

    def __init__(self, **kwargs):
        args = ApplyPatchArgs(**kwargs).model_dump()
        self.action = {"type": "apply_patch", **args}


class RunTests(EnvToolCall):
    """run_tests(tests?: list[str])"""

    def __init__(self, tests: Optional[List[str]] = None):
        args = RunTestsArgs(tests=tests).model_dump()
        self.action = {"type": "run_tests", **args}


class Submit(EnvToolCall):
    """
    Convenience "finish" action – identical to `RunTests()` but signals that the
    agent intends to terminate afterwards.  The learner/host can decide to end
    the episode once this tool is called.
    """

    def __init__(self, reason: str = "proposed fix ready"):
        self.action = {
            "type": "run_tests",
            "tests": None,
            "_submit": True,
            "reason": reason,
        }


# ───────────────────────────── Observation helper ────────────────────────────
class SynthSweBenchObservationCallable(GetObservationCallable):
    """
    Trivial observation wrapper – pretty-prints the keys that matter.
    Can be swapped out for something more sophisticated if desired.
    """

    async def get_observation(
        self, pub: Dict[str, Any], priv: Dict[str, Any]
    ) -> InternalObservation:
        out = {"last_reward": priv.get("reward_last", 0.0)}
        out.update(pub)  # include everything the engine exposed
        return out


# ───────────────────────────── Environment class ─────────────────────────────
class SweBenchEnvironment(StatefulEnvironment):
    """
    Bridges `SWEBenchEngine` → agent tools.

    * initialise → clones repo, returns initial observation
    * step       → executes exactly ONE `EnvToolCall` (wrapped action)
    * checkpoint → serialises engine for debugging / offline scoring
    """

    def __init__(
        self,
        task_instance,
        custom_obs: Optional[GetObservationCallable] = None,
    ) -> None:
        self.engine = SWEBenchEngine(task_instance.initial_engine_snapshot)  # type: ignore[arg-type]
        self.name = "SWE-bench-Env"
        self.custom_obs = custom_obs or SynthSweBenchObservationCallable()

    # ─────────────────── lifecycle ───────────────────
    async def initialize(self) -> InternalObservation:
        obs = self.engine._reset_engine()  # returns a *public* observation dict
        return await self._obs(obs)

    async def step(
        self,
        calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]],
    ) -> InternalObservation:
        # normalise shape → [[EnvToolCall]]
        if isinstance(calls, EnvToolCall):
            calls = [[calls]]
        elif calls and isinstance(calls[0], EnvToolCall):
            calls = [calls]

        action_dict = calls[0][0].action  # single-tool call per turn
        obs = self.engine._step_engine(action_dict)
        return await self._obs(obs)

    async def checkpoint(self) -> InternalObservation:
        return self.engine._serialize_engine()

    # ─────────────────── helpers ────────────────────
    async def _obs(self, pub: Dict[str, Any]) -> InternalObservation:
        """
        In this environment the engine already returns a *public* observation
        dict (no private view exposed to the agent).  We still pass it through
        an observation callable to keep parity with other envs.
        """
        return await self.custom_obs.get_observation(
            pub, {"reward_last": pub.get("reward", 0)}
        )

# Alias wrappers for compatibility with tests expecting OpenFile
OpenFile = ReadFile
OpenFileArgs = ReadFileArgs
