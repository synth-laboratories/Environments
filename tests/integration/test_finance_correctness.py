import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.examples.finance_agent import (
    FinanceEnv,
    create_finance_taskset,
    SubmitAnswerTool,
)


async def run_correct_incorrect():
    taskset = await create_finance_taskset()
    inst = taskset.instances[0]

    # correct answer scenario
    env = FinanceEnv(inst)
    await env.initialize()
    obs_correct = await env.step([[SubmitAnswerTool(answer=inst.metadata.answer)]])
    assert obs_correct["is_correct"] is True
    assert obs_correct["terminated"] is True

    # incorrect answer scenario
    env_wrong = FinanceEnv(inst)
    await env_wrong.initialize()
    obs_wrong = await env_wrong.step([[SubmitAnswerTool(answer="wrong answer")]])
    assert obs_wrong["is_correct"] is False
    assert obs_wrong["terminated"] is True


def test_finance_answer_evaluation():
    asyncio.run(run_correct_incorrect())
