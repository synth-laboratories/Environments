import sys
from pathlib import Path

import pytest

# Add the repo root "src" directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import minihack  # Ensure MiniHack registers its environments
pytest.importorskip("minihack.envs.room")

from src.examples.minihack.environment import MiniHackEnvironment
from src.environment.tools import EnvToolCall

@pytest.mark.asyncio
async def test_text_mode_step():
    env = MiniHackEnvironment()
    obs = await env.initialize()
    assert "board" in obs
    assert obs["step"] == 0
    action = EnvToolCall(tool="interact", args={"action": 0})
    obs = await env.step(action)
    assert obs["step"] == 1
    assert "reward_last" in obs

@pytest.mark.asyncio
async def test_vlm_mode_returns_image():
    env = MiniHackEnvironment(vlm_mode=True)
    await env.initialize()
    action = EnvToolCall(tool="interact", args={"action": 0})
    obs = await env.step(action)
    assert "board_image_b64" in obs
    assert isinstance(obs["board_image_b64"], str)
