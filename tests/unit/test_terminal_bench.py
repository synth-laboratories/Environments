import os
import sys
import asyncio
import pytest

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))
from examples.terminal_bench.environment import TerminalBenchEnvironment

class DummyResult:
    def model_dump(self):
        return {"ok": True}

class DummyHarness:
    def __init__(self, *a, **k):
        pass
    def run(self):
        return DummyResult()

def test_terminal_bench_environment_runs():
    env = TerminalBenchEnvironment(
        dataset_name="dummy",
        dataset_version="0",
        harness=DummyHarness(),
    )
    obs = asyncio.run(env.initialize())
    assert obs == {"initialized": True}
    result = asyncio.run(env.step(None))
    assert result["terminated"] is True
    assert result["results"] == {"ok": True}



