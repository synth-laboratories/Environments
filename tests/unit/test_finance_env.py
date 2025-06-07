import sys
from pathlib import Path
import asyncio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.examples.finance_agent import (
    FinanceEnv,
    create_finance_taskset,
    SubmitAnswerTool,
)


def test_finance_env_step():
    async def run():
        taskset = await create_finance_taskset()
        instance = taskset.instances[0]
        env = FinanceEnv(instance)
        obs = await env.initialize()
        assert "question" in obs
        step_obs = await env.step([[SubmitAnswerTool(answer=instance.metadata.answer)]])
        assert step_obs["is_correct"] is True
        assert step_obs["terminated"] is True

    asyncio.run(run())
