from __future__ import annotations

from typing import Optional, List, Union

from pydantic import BaseModel

from examples.minigrid.engine import (
    MiniGridEngine,
    MiniGridPublicState,
    MiniGridPrivateState,
)
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from src.environment.tools import (
    AbstractTool,
    EnvToolCall,
    ToolResult,
    TOOL_REGISTRY,
    register_tool,
)
from src.tasks.core import TaskInstance


# -------------------------------------------------------------
# Observation helpers
# -------------------------------------------------------------

OBJECT_TO_CHAR = {
    0: " ",  # unseen
    1: " ",  # empty
    2: "#",  # wall
    3: ".",  # floor
    4: "D",  # door
    5: "K",  # key
    6: "B",  # ball
    7: "X",  # box
    8: "G",  # goal
    9: "L",  # lava
    10: "A",  # agent
}


def image_to_text(image) -> str:
    rows = []
    for row in image:
        chars = [OBJECT_TO_CHAR.get(int(cell[0]), "?") for cell in row]
        rows.append("".join(chars))
    return "\n".join(rows)


class MiniGridTextObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: MiniGridPublicState, priv: MiniGridPrivateState
    ) -> InternalObservation:
        grid_text = image_to_text(pub.image)
        return {
            "grid": grid_text,
            "mission": pub.mission,
            "position": pub.agent_pos,
            "direction": pub.agent_dir,
            "step_count": pub.step_count,
            "max_steps": pub.max_steps,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }


class MiniGridVLMObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: MiniGridPublicState, priv: MiniGridPrivateState
    ) -> InternalObservation:
        return {
            "image": pub.image.tolist(),
            "mission": pub.mission,
            "position": pub.agent_pos,
            "direction": pub.agent_dir,
            "step_count": pub.step_count,
            "max_steps": pub.max_steps,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
        }


# -------------------------------------------------------------
# Tool definition
# -------------------------------------------------------------


class MiniGridActionInput(BaseModel):
    action: int


class MiniGridInteractTool(AbstractTool):
    name = "interact"
    description = "Perform an action in the MiniGrid environment"
    call_schema = MiniGridActionInput
    result_schema = ToolResult

    def __init__(self, engine: MiniGridEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated = self.call_schema(**call.args)
            priv, pub = await self.engine._step_engine(validated.action)
            return ToolResult(
                ok=True,
                payload={"public": pub, "private": priv},
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class MiniGridEnvironment(StatefulEnvironment):
    def __init__(
        self,
        task_instance: TaskInstance,
        text_obs: bool = True,
        custom_step_obs: Optional[GetObservationCallable] = None,
        custom_ckpt_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "MiniGrid"
        self.task_instance = task_instance
        if text_obs:
            self.custom_step_observation_callable = (
                custom_step_obs or MiniGridTextObservationCallable()
            )
            self.custom_checkpoint_observation_callable = (
                custom_ckpt_obs or MiniGridTextObservationCallable()
            )
        else:
            self.custom_step_observation_callable = (
                custom_step_obs or MiniGridVLMObservationCallable()
            )
            self.custom_checkpoint_observation_callable = (
                custom_ckpt_obs or MiniGridVLMObservationCallable()
            )
        self.engine = MiniGridEngine(task_instance)

        self._tool = MiniGridInteractTool(self.engine)
        if self._tool.name not in TOOL_REGISTRY:
            register_tool(self._tool)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def terminate(self) -> InternalObservation:
        priv = MiniGridPrivateState(
            reward_last=0.0,
            total_reward=self.engine._total_reward,
            terminated=True,
            truncated=False,
        )
        obs, _ = self.engine.env.reset()
        pub = self.engine._build_public_state(obs)
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    def validate_tool_calls(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> EnvToolCall:
        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Received empty list of tool calls")
            if isinstance(tool_calls[0], list):
                if not tool_calls[0]:
                    raise ValueError("Received empty inner list of tool calls")
                agent_call = tool_calls[0][0]
            else:
                agent_call = tool_calls[0]
        else:
            agent_call = tool_calls
        if not isinstance(agent_call, EnvToolCall):
            raise TypeError("tool_calls must contain EnvToolCall")
        if agent_call.tool != "interact":
            raise ValueError("Unknown tool: " + agent_call.tool)
        return agent_call

    async def step(
        self, tool_calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]]
    ) -> InternalObservation:
        agent_call = self.validate_tool_calls(tool_calls)
        result: ToolResult = await self._tool(agent_call)
        if not result.ok or not isinstance(result.payload, dict):
            raise RuntimeError(result.error or "Tool call failed")
        pub: MiniGridPublicState = result.payload["public"]
        priv: MiniGridPrivateState = result.payload["private"]
        return await self._to_observation(
            priv, pub, self.custom_step_observation_callable
        )

    async def checkpoint(self) -> InternalObservation:
        snapshot = await self.engine._serialize_engine()
        priv, pub = await self.engine._step_engine(0)  # no-op step for observation
        obs = await self._to_observation(
            priv, pub, self.custom_checkpoint_observation_callable
        )
        if isinstance(obs, dict):
            obs["engine_snapshot_data"] = snapshot.env_state
        return obs

    async def _to_observation(
        self,
        priv: MiniGridPrivateState,
        pub: MiniGridPublicState,
        obs_cb: Optional[GetObservationCallable],
    ) -> InternalObservation:
        active_cb = obs_cb or MiniGridTextObservationCallable()
        return await active_cb.get_observation(pub, priv)
