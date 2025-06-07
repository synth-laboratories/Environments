from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import gymnasium as gym
import minihack
import minihack.envs.room
from PIL import Image, ImageDraw, ImageFont
import io, base64

from src.stateful.engine import StatefulEngine
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from .taskset import MiniHackTaskInstance


def _ansi_to_base64(ansi: str) -> str:
    font = ImageFont.load_default()
    lines = ansi.splitlines()
    line_height = font.getbbox("A")[3]
    width = max(int(font.getlength(line)) for line in lines)
    img = Image.new("RGB", (width, line_height * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((0, i * line_height), line, font=font, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class MiniHackPublicState:
    board_text: str
    board_image: str
    step_count: int


@dataclass
class MiniHackPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool


class MiniHackTextObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: MiniHackPublicState, priv: MiniHackPrivateState
    ) -> InternalObservation:
        return {
            "board": pub.board_text,
            "step": pub.step_count,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }


class MiniHackVLMObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: MiniHackPublicState, priv: MiniHackPrivateState
    ) -> InternalObservation:
        obs = {
            "board": pub.board_text,
            "board_image_b64": pub.board_image,
            "step": pub.step_count,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }
        return obs


class MiniHackEngine(StatefulEngine):
    def __init__(self, task_instance: MiniHackTaskInstance):
        self.task_instance = task_instance
        self.env = gym.make(
            task_instance.env_id,
            observation_keys=("glyphs", "chars", "blstats"),
            render_mode="ansi",
        )
        self.total_reward = 0.0
        self.step_count = 0

    async def _reset_engine(
        self,
    ) -> Tuple[MiniHackPrivateState, MiniHackPublicState]:
        self.env.reset()
        self.total_reward = 0.0
        self.step_count = 0
        board = self.env.render()
        pub = MiniHackPublicState(
            board_text=board,
            board_image=_ansi_to_base64(board),
            step_count=self.step_count,
        )
        priv = MiniHackPrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
        )
        return priv, pub

    async def _step_engine(
        self, action: int
    ) -> Tuple[MiniHackPrivateState, MiniHackPublicState]:
        _, reward, terminated, truncated, _ = self.env.step(action)
        self.total_reward += float(reward)
        self.step_count += 1
        board = self.env.render()
        priv = MiniHackPrivateState(
            reward_last=float(reward),
            total_reward=self.total_reward,
            terminated=terminated,
            truncated=truncated,
        )
        pub = MiniHackPublicState(
            board_text=board,
            board_image=_ansi_to_base64(board),
            step_count=self.step_count,
        )
        return priv, pub
