from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from src.environment.shared_engine import InternalObservation
from src.stateful.core import StatefulEnvironment


class TerminalBenchEnvironment(StatefulEnvironment):
    """Wrapper around terminal-bench Harness for simple evaluation."""

    def __init__(
        self,
        dataset_name: str,
        dataset_version: str,
        *,
        agent_name: str = "terminus",
        model_name: str | None = None,
        output_path: str = "tb_runs",
        run_id: str = "synth-run",
        n_concurrent_trials: int = 1,
        n_attempts: int = 1,
        harness: Any | None = None,
    ) -> None:
        self.name = "TerminalBench"

        if harness is None:
            try:
                from terminal_bench import Harness  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime import
                raise ImportError(
                    "terminal-bench>=0.2.3 is required for TerminalBenchEnvironment"
                ) from exc

            harness = Harness(
                output_path=Path(output_path),
                run_id=run_id,
                agent_name=agent_name,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                model_name=model_name,
                n_concurrent_trials=n_concurrent_trials,
                n_attempts=n_attempts,
            )

        self._harness = harness
        self._results = None

    async def initialize(self) -> InternalObservation:  # type: ignore[override]
        return {"initialized": True}

    async def terminate(self) -> InternalObservation:  # type: ignore[override]
        return {"terminated": True, "results": self._serialize_results()}

    def validate_tool_calls(self, tool_calls: Any) -> None:  # type: ignore[override]
        if tool_calls:
            raise ValueError("TerminalBenchEnvironment does not accept tool calls")

    async def step(self, tool_calls: Any) -> InternalObservation:  # type: ignore[override]
        self.validate_tool_calls(tool_calls)
        self._results = await asyncio.to_thread(self._harness.run)
        return {"terminated": True, "results": self._serialize_results()}

    async def checkpoint(self) -> InternalObservation:  # type: ignore[override]
        return {"results": self._serialize_results()}

    def _serialize_results(self) -> Optional[dict]:
        if self._results is None:
            return None
        try:
            return self._results.model_dump()  # type: ignore[attr-defined]
        except Exception:
            return None
