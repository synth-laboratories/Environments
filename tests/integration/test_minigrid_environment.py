import pytest

from src.examples.minigrid.environment import MiniGridEnvironment
from src.examples.minigrid.taskset import DEFAULT_TASK_INSTANCE
from src.environment.tools import EnvToolCall


@pytest.mark.asyncio
async def test_minigrid_environment_text_obs():
    env = MiniGridEnvironment(DEFAULT_TASK_INSTANCE, text_obs=True)
    obs = await env.initialize()
    assert "grid" in obs
    assert isinstance(obs["grid"], str)

    step_obs = await env.step(EnvToolCall(tool="interact", args={"action": 0}))
    assert "grid" in step_obs

    term_obs = await env.terminate()
    assert term_obs["terminated"]


@pytest.mark.asyncio
async def test_minigrid_environment_vlm_obs():
    env = MiniGridEnvironment(DEFAULT_TASK_INSTANCE, text_obs=False)
    obs = await env.initialize()
    assert "image" in obs
    assert isinstance(obs["image"], list)

    step_obs = await env.step(EnvToolCall(tool="interact", args={"action": 0}))
    assert "image" in step_obs

    cp_obs = await env.checkpoint()
    assert "engine_snapshot_data" in cp_obs
    await env.terminate()
