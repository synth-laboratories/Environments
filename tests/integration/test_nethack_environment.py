import sys
from pathlib import Path
import asyncio
import nle.nethack as nh
import pytest

# Ensure src is on path
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "src"))
sys.path.append(str(repo_root))

from examples.nethack_le.environment import NethackEnvironment
from examples.nethack_le.taskset import INSTANCE


def test_nethack_text_mode_movement():
    async def run():
        env = NethackEnvironment(INSTANCE, mode="text")
        obs = await env.initialize()
        assert "screen_text" in obs
        assert not obs["terminated"]
        init_step = obs["step"]

        move_east = nh.ACTIONS.index(nh.CompassDirection.E)
        obs2 = await env.step({"tool": "nethack_action", "args": {"action": move_east}})
        assert obs2["step"] == init_step + 1
        assert not obs2["terminated"]

        await env.terminate()

    asyncio.run(run())


def test_nethack_image_mode_glyphs():
    async def run():
        env = NethackEnvironment(INSTANCE, mode="image")
        obs = await env.initialize()
        assert "glyphs" in obs
        move_north = nh.ACTIONS.index(nh.CompassDirection.N)
        obs2 = await env.step(
            {"tool": "nethack_action", "args": {"action": move_north}}
        )
        assert obs2["step"] == 1
        await env.terminate()

    asyncio.run(run())
