# examples/swe_bench/react_agent.py
"""
ReAct-style agent for SWE-bench / SWE-Gym
─────────────────────────────────────────
*   Uses **function-calling** tools that the `SweBenchEnvironment`
    exposes (`read_file`, `apply_patch`, `run_tests`, `submit`).
*   Keeps an abbreviated *tool history* so the LLM can reason
    over what it has already tried.
*   Very small prompt – meant as a scaffold you can iterate on.
"""

from __future__ import annotations

import json, uuid, textwrap, asyncio
from collections import deque
from typing import Any, Dict, Deque, List, Optional

from pydantic import BaseModel, Field

from synth_ai.zyk import LM
from environment.tools import EnvToolCall
from synth_sdk.tracing.utils import get_system_id

from examples.swe_bench.environment import (
    SweBenchEnvironment,
    ReadFile,     ReadFileArgs,
    ApplyPatch,   ApplyPatchArgs,
    RunTests,     RunTestsArgs,
    Submit,       SubmitArgs,
)

# ─────────────────────────── Terminate surrogate (optional) ──────────────────
class TerminateArgs(BaseModel):
    reason: str

# ───────────────────────────── ReAct Agent class ─────────────────────────────
class ReActSweBenchAgent:
    """
    Minimal ReAct agent – emits exactly **one** tool-call each `act` turn.
    """

    def __init__(self, llm: LM, *, max_steps: int = 15, tool_window: int = 20):
        self.llm, self.max_steps = llm, max_steps

        self.memory: Deque[Dict[str, Any]] = deque(maxlen=30)      # observation history
        self.tool_hist: Deque[Dict[str, Any]] = deque(maxlen=tool_window)

        self.system_name, self.system_instance_id = "swe-react", str(uuid.uuid4())

        # ---------- function-calling schema advertised to the LLM ----------
        self.tools = [
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a slice of a repository file",
                "parameters": ReadFileArgs.model_json_schema(),
            }},
            {"type": "function", "function": {
                "name": "apply_patch",
                "description": "Apply a unified-diff patch to the workspace",
                "parameters": ApplyPatchArgs.model_json_schema(),
            }},
            {"type": "function", "function": {
                "name": "run_tests",
                "description": "Run pytest (default = failing+passing tests of this task)",
                "parameters": RunTestsArgs.model_json_schema(),
            }},
            {"type": "function", "function": {
                "name": "submit",
                "description": "Run full tests and finish the episode",
                "parameters": SubmitArgs.model_json_schema(),
            }},
            {"type": "function", "function": {             # optional early stop
                "name": "terminate",
                "description": "Give up / stop early",
                "parameters": TerminateArgs.model_json_schema(),
            }},
        ]

    # ────────────────────────────── core loop ──────────────────────────────
    async def act(self, obs: Dict[str, Any]) -> EnvToolCall:
        """
        Decide on one tool call given the latest environment observation.
        """
        # hide large blobs from the prompt
        obs_short = {k: v for k, v in obs.items() if k not in ("file_snippet", "test_output")}
        self.memory.append({"obs": obs_short})

        # quick summaries for context
        last_action  = obs.get("last_action", "reset")
        tests_passed = obs.get("tests_passed")
        reward       = obs.get("reward", 0.0)

        # tool history as a bullet list
        hist_lines = []
        for t in list(self.tool_hist)[-10:]:
            name = t["name"]
            detail = (
                t["args"].get("path")            if name == "read_file" else
                t["args"].get("patch","")[:30]   if name == "apply_patch" else
                ""
            )
            hist_lines.append(f"• {name} {detail}")
        hist_block = "\n".join(hist_lines) or "None yet."

        system_prompt = textwrap.dedent(f"""
            You are an autonomous software-bug-fixing agent.

            Tools:
              • read_file(path,start?,end?)
              • apply_patch(patch)
              • run_tests()
              • submit(reason)

            Guidelines:
              • Think step-by-step.  Use read_file to inspect code,
                apply_patch with *small* diffs, then run_tests.
              • Call submit once all failing tests pass.
              • Max {self.max_steps} steps.

            Recent tool calls:
            {hist_block}
        """).strip()

        user_prompt = textwrap.dedent(f"""
            Last env action : {last_action}
            Tests passed    : {tests_passed}
            Reward Δ        : {reward}

            What tool will you call next?
        """).strip()

        resp = await self.llm.respond_async(
            system_message=system_prompt,
            user_message=user_prompt,
            tools=self.tools,
        )

        if not resp.tool_calls:           # fall-back → run_tests
            return RunTests()

        # We take only *first* tool call each turn
        tc = resp.tool_calls[0]
        fn_name   = tc.function.name
        args_json = tc.function.arguments
        try:
            args_dict = json.loads(args_json)
        except Exception:
            args_dict = {}

        # ---------- translate → EnvToolCall ----------
        if fn_name == "read_file":
            call = ReadFile(**ReadFileArgs(**args_dict).model_dump())
        elif fn_name == "apply_patch":
            call = ApplyPatch(**ApplyPatchArgs(**args_dict).model_dump())
        elif fn_name == "run_tests":
            call = RunTests(**RunTestsArgs(**args_dict).model_dump())
        elif fn_name == "submit":
            call = Submit(**SubmitArgs(**args_dict).model_dump())
        else:                          # terminate / unknown
            call = Submit(reason="terminate")

        # record concise history entry
        self.tool_hist.append({
            "name": fn_name,
            "args": args_dict,
        })
        return call


# ────────────────────────────── quick smoke-test ─────────────────────────────
if __name__ == "__main__":                       # pragma: no cover
    from examples.swe_bench.taskset import create_taskset
    import asyncio

    async def _demo():
        # load first lite task
        ts  = await create_taskset(dataset="swe-bench", config="lite")
        inst = ts.instances[0]

        env  = SweBenchEnvironment(inst)
        llm  = LM(model_name="gpt-4.1-nano",
                  formatting_model_name="gpt-4.1-nano", temperature=0.0)
        agent= ReActSweBenchAgent(llm)

        obs  = await env.initialize()
        for step in range(agent.max_steps):
            tool_call = await agent.act(obs)
            obs       = await env.step(tool_call)
            print(f"[{step}] {tool_call.action['type']} → reward={obs.get('reward')}")
            if obs.get("terminated"):
                break
        print("Episode finished.")

    asyncio.run(_demo())