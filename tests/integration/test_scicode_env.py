import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import asyncio
from uuid import uuid4

import pytest

from examples.scicode.environment import SciCodeEnv
from examples.scicode.schema import SciCodeTaskInstance, SciCodeTaskInstanceMetadata
from examples.scicode.tools import SubmitAnswerTool
from examples.scicode.engine import load_tasks as original_load_tasks
from examples.scicode import engine as scicode_engine
from src.tasks.core import Impetus, Intent


@pytest.fixture(autouse=True)
def patch_load_tasks(monkeypatch):
    tasks = [
        {
            "id": "1",
            "prompt": "What is 2 + 2?",
            "solution": "4",
            "category": "math",
            "difficulty": "easy",
        }
    ]
    monkeypatch.setattr(scicode_engine, "load_tasks", lambda: tasks)
    yield tasks
    monkeypatch.setattr(scicode_engine, "load_tasks", original_load_tasks)


def make_instance(task):
    metadata = SciCodeTaskInstanceMetadata(
        category=task["category"],
        difficulty=task["difficulty"],
        solution=task["solution"],
    )
    impetus = Impetus(instructions="Solve the SciCode task.")
    intent = Intent(rubric={}, gold_trajectories=None, gold_state_diff={})
    return SciCodeTaskInstance(
        id=uuid4(),
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )


@pytest.mark.asyncio
async def test_scicode_env_correct_answer(patch_load_tasks):
    task = patch_load_tasks[0]
    env = SciCodeEnv(make_instance(task))
    obs = await env.initialize()
    assert obs["prompt"] == task["prompt"]
    obs = await env.step([[SubmitAnswerTool(answer="4")]])
    assert obs["is_correct"] is True
    assert obs["terminated"] is True


@pytest.mark.asyncio
async def test_scicode_env_wrong_answer(patch_load_tasks):
    task = patch_load_tasks[0]
    env = SciCodeEnv(make_instance(task))
    await env.initialize()
    obs = await env.step([[SubmitAnswerTool(answer="5")]])
    assert obs["is_correct"] is False
    assert obs["terminated"] is True
