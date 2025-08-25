import sys
from pathlib import Path
import pytest

# Ensure the repo's src directory is importable regardless of pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.examples.treasure_hunter.environment import TreasureHunterEnvironment
from src.examples.treasure_hunter.taskset import INSTANCE
from src.environment.tools import EnvToolCall

@pytest.mark.asyncio
async def test_treasure_hunter_episode():
    env = TreasureHunterEnvironment(INSTANCE)
    obs = await env.initialize()
    assert obs["position"] == (0, 0)
    # Move east 4 times and south 4 times to reach treasure
    for _ in range(4):
        obs = await env.step(EnvToolCall(tool="command", args={"command": "east"}))
    for _ in range(4):
        obs = await env.step(EnvToolCall(tool="command", args={"command": "south"}))
    assert obs["position"] == (4, 4)
    assert not obs["terminated"]
    obs = await env.step(EnvToolCall(tool="command", args={"command": "take"}))
    assert obs["terminated"]
    assert obs["has_treasure"] is True
    assert obs["total_reward"] == pytest.approx(1.0)
