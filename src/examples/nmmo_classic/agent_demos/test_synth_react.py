# examples/nmmo3/test_synth_react.py
"""
Minimal ReAct-style agent + evaluation harness for Neural MMO 3 wrapped
through PufferLib (see examples/nmmo3/engine.py and environment.py).

• One-shot unit-test (`test_react_agent_nmmo`) – validates the plumbing.
• Quick-n-dirty eval (`eval_react_nmmo`)  – runs 2×(easy, hard) seeds and
  prints success / average-score table.

Success is defined as "episode terminates (death or tick-limit) without
crashing, and the agent earned any positive reward".
"""

from __future__ import annotations
import asyncio, json, uuid, logging, random, pytest
from typing import Dict, Any, List, Optional, Deque

from dataclasses import asdict
from collections import deque
from pathlib import Path
from pydantic import BaseModel, Field

# --- LLM / tracing stubs ----------------------------------------------------
from synth_ai.zyk import LM
from synth_sdk.tracing.decorators import trace_event_async
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_sdk.tracing.utils import get_system_id
# ---------------------------------------------------------------------------

from examples.nmmo_classic.environment import NMMO3Environment
from examples.nmmo_classic.state import NMMO3PublicState, NMMO3PrivateState
from examples.nmmo_classic.taskset import (
    NMMO3TaskInstance,
    NMMO3TaskInstanceMetadata,
)

from tasks.core import Impetus, Intent
from environment.tools import EnvToolCall
from environment.shared_engine import GetObservationCallable, InternalObservation

logging.disable(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# Action helpers
# ─────────────────────────────────────────────────────────────────────────────
#
# We expose *five* high-level aliases the LLM can pick from.
# Each converts to the Dict expected by PufferLib/NMMO.
#
MOVE_DIRS = ["north", "south", "west", "east", "center"]
ACTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "move_north": {"Move": {"direction": 0}},
    "move_south": {"Move": {"direction": 1}},
    "move_west": {"Move": {"direction": 2}},
    "move_east": {"Move": {"direction": 3}},
    "attack_melee": {"Attack": {"style": 0, "target": 0}},  # target = self (no-op)
    "eat_slot0": {"Eat": {"inventory": 0}},
    "drink_slot0": {"Drink": {"inventory": 0}},
    "pickup": {"Pickup": {"pickup": 0}},
    "attack_range": {"Attack": {"style": 1, "target": 0}},
    "attack_mage": {"Attack": {"style": 2, "target": 0}},
    "gather_fishing": {"Gather": {"skill": 0}},
    "gather_hunting": {"Gather": {"skill": 1}},
    "gather_prospecting": {"Gather": {"skill": 2}},
    "gather_carving": {"Gather": {"skill": 3}},
    "gather_alchemy": {"Gather": {"skill": 4}},
    "buy_slot0": {"Buy": {"item": 0}},
    "sell_slot0": {"Sell": {"inventory": 0}},
    "token": {"Token": {"words": [0, 0, 0, 0, 0, 0, 0, 0]}},
    "noop": {},  # empty Dict = no-op
}

# The reverse mapping is trivial (string→dict only), so we skip INT→STR.


# ─────────────────────────────────────────────────────────────────────────────
# LL-M-friendly observation formatting
# ─────────────────────────────────────────────────────────────────────────────
def format_for_llm(pub: NMMO3PublicState, priv: NMMO3PrivateState) -> str:
    inv_str = ", ".join(f"{k}:{v}" for k, v in pub.inventory.items() if v) or "empty"
    return (
        f"Tick {pub.tick} / {pub.max_episode_steps}\n"
        f"Pos: {pub.position} Facing:{pub.facing}\n"
        f"HP:{pub.health}  Stam:{pub.stamina}\n"
        f"Inv:[{inv_str}]\n"
        f"Last r:{priv.reward_last_step:.2f}  Tot r:{priv.total_reward_episode:.1f}\n"
        f"Terminated:{priv.terminated}  Truncated:{priv.truncated}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observation callable that keeps *one* history entry (for demo)
# ─────────────────────────────────────────────────────────────────────────────
class NMMOHistoryObs(GetObservationCallable):
    def __init__(self, max_hist: int = 1):
        self._hist: Deque[str] = deque(maxlen=max_hist)

    async def get_observation(
        self, pub: NMMO3PublicState, priv: NMMO3PrivateState
    ) -> InternalObservation:
        """
        Return a flat InternalObservation containing all public/private fields,
        plus the formatted text and history list.
        """
        txt = format_for_llm(pub, priv)
        self._hist.append(txt)
        # flatten fields
        return {
            "tick": pub.tick,
            "num_steps_taken": pub.num_steps_taken,
            "max_episode_steps": pub.max_episode_steps,
            "position": pub.position,
            "facing": pub.facing,
            "health": pub.health,
            "stamina": pub.stamina,
            "inventory": pub.inventory,
            "local_terrain": pub.local_terrain,
            "visible_entities": pub.visible_entities,
            "team_score": pub.team_score,
            "personal_score": pub.personal_score,
            "reward_last": priv.reward_last_step,
            "total_reward": priv.total_reward_episode,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
            "formatted": txt,
            "history": list(self._hist),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic tool-schemas
# ─────────────────────────────────────────────────────────────────────────────
class NMMOInteractArgs(BaseModel):
    action_name: str = Field(description=f"One of {list(ACTION_TEMPLATES.keys())}")
    reasoning: str = Field(description="Why this action?")


class TerminateArgs(BaseModel):
    reason: str


# tiny wrapper so evaluation code can use the standard EnvToolCall list-of-list
class Act(EnvToolCall):
    def __init__(self, action_dict: Dict[str, Any]):
        self.action = action_dict


# ─────────────────────────────────────────────────────────────────────────────
# ReAct agent
# ─────────────────────────────────────────────────────────────────────────────
class ReActAgent:
    def __init__(self, llm, max_turns: int = 50):
        self.llm, self.max_turns = llm, max_turns
        self.history: List[Dict[str, Any]] = []
        self.system_name = "nmmo-react-ex"
        self.system_id = get_system_id(self.system_name)
        self.system_instance_id = str(uuid.uuid4())

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "nmmo_interact",
                    "description": "Send a single high-level action to the env",
                    "parameters": NMMOInteractArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "Stop the episode",
                    "parameters": TerminateArgs.model_json_schema(),
                },
            },
        ]

    # prompt helper
    def _prompt_history(self) -> str:
        parts = []
        for h in self.history:
            if h["type"] == "obs":
                parts.append(f"OBS:\n{h['content']}")
            elif h["type"] == "tool":
                parts.append(f"THOUGHT+ACTION:\n{h['name']} {json.dumps(h['args'])}")
            elif h["type"] == "resp":
                parts.append("TOOL_RESPONSE: OK")
        return "\n".join(parts)

    # main policy -----------------------------------------------------------
    @trace_event_async(event_type="react_agent_decide")
    async def decide(self, obs_txt: str) -> Dict[str, Any] | str:
        self.history.append({"type": "obs", "content": obs_txt})

        sys_msg = (
            "You are an agent in Neural MMO. Survive and maximise reward.\n"
            f"Use one of these actions: {list(ACTION_TEMPLATES.keys())}.\n"
            "You MUST respond with exactly one tool call."
        )
        user_msg = (
            self._prompt_history()
            + "\n\nGiven the last observation, what should you do next?"
        )
        resp = await self.llm.respond_async(
            system_message=sys_msg, user_message=user_msg, tools=self.tools
        )
        tool_call = resp.tool_calls[0] if resp.tool_calls else None
        if not tool_call:
            # fallback
            return {"nmmo_interact": {"action_name": "noop", "reasoning": "fallback"}}

        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        self.history.append({"type": "tool", "name": name, "args": args})
        return {name: args}


# ─────────────────────────────────────────────────────────────────────────────
# pytest-style smoke-test
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_react_agent_nmmo(tmp_path: Path):
    # 1. build a *very* small task-instance
    meta = NMMO3TaskInstanceMetadata(
        difficulty="easy",
        seed=123,
        map_size=64,
        season="summer",
        resource_density=0.0,
        water_pct=0.0,
        hostiles_25=0,
        forage_tiles_25=0,
        spawn_biome="plains",
    )
    inst = NMMO3TaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="explore"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=meta,
        is_reproducible=True,
        initial_engine_snapshot=None,
        config={"seed": 123, "map_size": 64, "tick_limit": 200},
    )

    obs_cb = NMMOHistoryObs()
    env = NMMO3Environment(inst, custom_step_obs=obs_cb)

    llm = LM(
        model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0
    )
    agent = ReActAgent(llm, max_turns=10)

    # initialise
    payload = await env.initialize()
    obs_txt = payload["formatted"]
    priv = payload  # flat observation

    for _ in range(agent.max_turns):
        call_dict = await agent.decide(obs_txt)

        if "terminate" in call_dict:
            break

        action_name = call_dict["nmmo_interact"]["action_name"]
        action_dict = ACTION_TEMPLATES.get(action_name, {})
        step_payload = await env.step([[Act(action_dict)]])
        obs_txt = step_payload["formatted"]
        priv = step_payload  # flat observation
        agent.history.append({"type": "resp"})

        if priv["terminated"] or priv["truncated"]:
            break

    assert priv["total_reward"] >= 0.0  # smoke-criterion


# ─────────────────────────────────────────────────────────────────────────────
# simple evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
async def eval_react_nmmo() -> None:
    """
    Run 2 seeds for 'easy' and 'hard', print success / average reward.
    """

    async def make_instance(seed: int, map_size: int, diff: str) -> NMMO3TaskInstance:
        meta = NMMO3TaskInstanceMetadata(
            difficulty=diff,
            seed=seed,
            map_size=map_size,
            season="summer",
            resource_density=0.0,
            water_pct=0.0,
            hostiles_25=0,
            forage_tiles_25=0,
            spawn_biome="plains",
        )
        return NMMO3TaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(instructions="play"),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=meta,
            is_reproducible=True,
            initial_engine_snapshot=None,
            config={"seed": seed, "map_size": map_size, "tick_limit": 500},
        )

    configs = [("easy", 64, 2), ("hard", 128, 2)]
    table = []

    for label, msize, n in configs:
        seeds = [100 + i for i in range(n)]
        insts = [await make_instance(s, msize, label) for s in seeds]
        results = []

        for inst in insts:
            obs_cb = NMMOHistoryObs()
            env = NMMO3Environment(inst, custom_step_obs=obs_cb)
            llm = LM(
                model_name="gpt-4.1-nano",
                formatting_model_name="gpt-4.1-nano",
                temperature=0.0,
            )
            agent = ReActAgent(llm, max_turns=30)
            p = await env.initialize()
            txt = p["formatted"]
            priv = p  # flat observation
            for _ in range(agent.max_turns):
                call = await agent.decide(txt)
                if "terminate" in call:
                    break
                act_dict = ACTION_TEMPLATES.get(
                    call["nmmo_interact"]["action_name"], {}
                )
                p = await env.step([[Act(act_dict)]])
                txt = p["formatted"]
                priv = p  # flat observation
                agent.history.append({"type": "resp"})
                if priv["terminated"] or priv["truncated"]:
                    break
            results.append(priv["total_reward"])

        success = sum(r > 0 for r in results)
        avg_r = sum(results) / len(results)
        table.append([label, f"{success}/{len(results)}", f"{avg_r:.1f}"])

    from tabulate import tabulate

    print(
        tabulate(
            table,
            headers=["Difficulty", "Pos-reward runs", "Avg reward"],
            tablefmt="github",
        )
    )


# quick CLI
if __name__ == "__main__":
    asyncio.run(eval_react_nmmo())
