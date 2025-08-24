"""ReAct agent demo for MiniGrid environment."""

import asyncio
import time
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import SynthAI LM and BaseTool (migrated to horizons.lm)
from horizons.lm import LM
from horizons.lm.tools.base import BaseTool
from horizons.tracing_v3 import SessionTracer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from horizons.environments.environment.tools import EnvToolCall
from horizons.environments.examples.minigrid.environment import MiniGridEnvironment
from horizons.environments.examples.minigrid.taskset import (
    DEFAULT_MINIGRID_TASK,
    create_minigrid_taskset,
)


# --- Pydantic Models for Tool Arguments ---
class MiniGridActArgs(BaseModel):
    """Arguments for MiniGrid action sequence."""

    actions_list: List[str] = Field(
        description="A list of 1-10 actions to execute in order. Each must be one of: 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'"
    )
    reasoning: str = Field(description="Brief explanation for why these actions were chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""

    reason: str = Field(description="Reason for termination")


# --- Tool Definitions ---


class MiniGridActTool(BaseTool):
    """Tool for performing an action in MiniGrid."""

    name: str = "minigrid_act"
    arguments: type[BaseModel] = MiniGridActArgs
    description: str = (
        "Perform a short sequence (1-10) of actions in MiniGrid; actions are executed consecutively."
    )


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""

    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- ReAct Agent ---
class MiniGridReActAgent:
    """ReAct agent for MiniGrid environments."""

    def __init__(self, llm: LM, max_turns: int = 30, verbose: bool = False, tracer: SessionTracer | None = None):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.debug_log = []  # Store all prompts and responses for debugging
        self.system_name: str = "minigrid-react-agent"  # Required for synth-sdk tracing
        self.system_instance_id: str = str(uuid.uuid4())  # Required for synth-sdk tracing
        self.tracer: SessionTracer | None = tracer

        # Available tools
        self.tools = [MiniGridActTool(), TerminateTool()]

    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for LLM."""
        if "observation" in obs:
            return obs["observation"]

        # Fallback formatting
        parts = []
        if "mission" in obs:
            parts.append(f"Mission: {obs['mission']}")
        if "terminated" in obs:
            parts.append(f"Terminated: {obs['terminated']}")
        if "reward_last" in obs:
            parts.append(f"Last Reward: {obs['reward_last']:.3f}")
        if "total_reward" in obs:
            parts.append(f"Total Reward: {obs['total_reward']:.3f}")

        return "\n".join(parts)

    async def decide(self, obs: str, task_description: str, turn: int) -> Dict[str, Any]:
        """Get LLM decision for next action."""
        system_message = f"""You are playing a MiniGrid environment. {task_description}

CRITICAL UNDERSTANDING OF THE GRID:

1. HOW TO READ THE GRID:
   - The grid shows a top-down view of a small world
   - Your position is shown by an arrow: → ↓ ← ↑
   - The arrow shows both WHERE you are and WHICH DIRECTION you're facing

2. GRID SYMBOLS:
   - → ↓ ← ↑ = YOU (the arrow points in the direction you're facing)
   - # = wall (CANNOT move through these)
   - . = empty space (CAN move through these)
   - G = goal (your target - GET HERE to win!)
   - L = lava (AVOID - stepping on this ends the game)
   - K = key, D = door, B = ball (for special levels)
   - ? = edge of the grid (CANNOT move here - it's the boundary)

3. HOW MOVEMENT WORKS:
   - 'forward' = move ONE space in the direction your arrow is pointing
   - 'left' = turn 90 degrees left (changes arrow direction, doesn't move you)
   - 'right' = turn 90 degrees right (changes arrow direction, doesn't move you)
   - You CANNOT move through walls (#) or boundaries (?)
   
4. DEBUG MESSAGES:
   - "Forward blocked by wall" = you tried to move into a wall
   - "Forward blocked by boundary" = you tried to move outside the grid
   - "Moved forward" = you successfully moved

5. IMPORTANT - LIMITED VISIBILITY:
   - You have LIMITED VISION and can only see a small area around you
   - The goal (G) might NOT be visible initially - you need to EXPLORE
   - The ? symbols show areas beyond your current view
   - You must move around the maze to discover new areas

6. EXPLORATION STRATEGY:
   - If you DON'T see the goal (G), you must EXPLORE the maze
   - Move systematically through empty spaces (.) to reveal new areas
   - Try to explore unexplored paths rather than revisiting the same spots
   - Keep track of where you've been to avoid going in circles
   - When you discover the goal (G), then plan a path to reach it

7. TOOL CALL FORMAT (MANDATORY):
   - Call the tool `minigrid_act` ONCE per turn with an `actions_list` field containing 1-5 actions
   - Example: {{"tool": "minigrid_act", "args": {{"actions_list": ["right", "forward", "forward"]}}}}

8. LEARN FROM PAST ACTIONS:
   - If an action was blocked, DON'T repeat it immediately
   - If you keep getting blocked moving forward, try turning left or right
   - If you're stuck in a pattern, break it by trying a different approach"""

        # Extract debug information to highlight it
        debug_info = ""
        if "Debug:" in obs:
            debug_lines = [
                line
                for line in obs.split("\n")
                if "Debug:" in line or "Last action result:" in line
            ]
            if debug_lines:
                debug_info = (
                    "\n\n🚨 IMPORTANT DEBUG INFORMATION:\n"
                    + "\n".join(f"• {line}" for line in debug_lines)
                    + "\n"
                )

        # Build action history string including structured debug hints from env
        action_history = ""
        if len(self.history) > 0:
            action_history = "\n\nRECENT HISTORY (Last 3 Actions):\n"
            for i, h in enumerate(self.history[-3:], 1):
                action_history += f"{i}. {h}\n"
            action_history += "\nBased on this history, avoid repeating failed actions and learn from what worked!\n"

        user_content = (
            f"Current state:\n{obs}{debug_info}{action_history}"
            "\nCRITICAL: Check the debug information above! If blocked by wall, you MUST turn or try a different action."
            "\n\nPropose 1-5 actions to execute next, as a single tool call to 'minigrid_act' with 'actions_list'."
            " Choose exploration-friendly moves when the goal isn't visible."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

        # Log the prompt
        prompt_entry = {
            "turn": turn,
            "type": "prompt",
            "messages": messages,
            "tools": self.tools,
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(prompt_entry)

        # Measure LLM latency
        _t0 = time.perf_counter()
        response = await self.llm.respond_async(messages=messages, tools=self.tools)
        llm_latency_ms = int((time.perf_counter() - _t0) * 1000)

        # Log the response
        response_entry = {
            "turn": turn,
            "type": "llm_response",
            "response": str(response),
            "response_type": type(response).__name__,
            "tool_calls": getattr(response, "tool_calls", None),
            "content": getattr(response, "content", None),
            "timestamp": datetime.now().isoformat(),
            "llm_latency_ms": llm_latency_ms,
        }
        self.debug_log.append(response_entry)

        # Print and trace latency
        if self.verbose:
            print(f"LLM call latency: {llm_latency_ms} ms")
        try:
            if self.tracer:
                await self.tracer.record_message(
                    json.dumps({"turn": turn, "llm_latency_ms": llm_latency_ms}),
                    "system",
                    metadata={"subtype": "llm_latency"},
                )
        except Exception:
            pass

        # Debug: Print response type
        if self.verbose:
            print(f"DEBUG: LLM response type: {type(response)}")
            print(f"DEBUG: LLM response full: {response}")
            if hasattr(response, "tool_calls"):
                print(f"DEBUG: Tool calls: {response.tool_calls}")
                if response.tool_calls:
                    print(f"DEBUG: First tool call: {response.tool_calls[0]}")
                    print(f"DEBUG: First tool call type: {type(response.tool_calls[0])}")
            if hasattr(response, "content"):
                print(f"DEBUG: Response content: {response.content}")

        # Parse tool calls - fail fast, but support both dict/object tool_call formats
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            # Handle different response formats
            if isinstance(tool_call, dict):
                # Dict format from LLM
                func = tool_call["function"]
                action = {
                    "name": func["name"],
                    "parameters": json.loads(func["arguments"]),
                }
            elif hasattr(tool_call, "function"):
                # Object format
                action = {
                    "name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments),
                }
            else:
                # Unexpected format - fail fast
                raise ValueError(f"Unexpected tool_call format: {tool_call}")
        else:
            # No tool call - fail fast
            raise ValueError("No tool call returned from LLM")

        # Log the parsed action
        action_entry = {
            "turn": turn,
            "type": "parsed_action",
            "action": action,
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(action_entry)

        return action

    async def run_episode(self, env: MiniGridEnvironment) -> Dict[str, Any]:
        """Run one episode in the environment."""
        # Initialize tracer (v3)
        tracer = self.tracer or SessionTracer()
        await tracer.initialize()

        # Initialize
        obs = await env.initialize()
        task_description = env.task_instance.impetus.instructions

        if self.verbose:
            print(f"\nTask: {task_description}")
            print(f"Initial observation:\n{self._format_observation(obs)}\n")

        success = False
        total_reward = 0.0
        last_reward = 0.0

        should_break = False
        async with tracer.session(metadata={
            "system_name": self.system_name,
            "system_instance_id": self.system_instance_id,
            "env_name": getattr(env.engine, "env_name", None),
            "seed": getattr(env.engine, "seed", None),
            "difficulty": getattr(env.engine, "difficulty", None),
        }):
            for turn in range(self.max_turns):
                async with tracer.timestep(step_id=f"turn_{turn+1}", turn_number=turn + 1):
                    # Format observation
                    formatted_obs = self._format_observation(obs)

                    # Log the observation
                    obs_entry = {
                        "turn": turn,
                        "type": "observation",
                        "raw_obs": obs,
                        "formatted_obs": formatted_obs,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.debug_log.append(obs_entry)
                    try:
                        await tracer.record_message(
                            formatted_obs, "system", metadata={"subtype": "observation"}
                        )
                    except Exception:
                        pass

                    # Get agent decision
                    action = await self.decide(formatted_obs, task_description, turn)

                    if self.verbose:
                        print(f"\nTurn {turn + 1}:")
                        print(f"Action: {action['name']}")
                        if "parameters" in action:
                            print(f"Parameters: {action['parameters']}")

                    # Check for termination
                    if action["name"] == "terminate":
                        if self.verbose:
                            print(f"Agent terminated: {action['parameters']['reason']}")
                        break

                    # Execute: map the agent's actions_list to environment 'actions' arg
                    args = action["parameters"]
                    if "actions_list" in args:
                        env_args = {"actions": args["actions_list"]}
                    else:
                        # Backward compatibility if model returns single 'action'
                        if "action" in args:
                            env_args = {"action": args["action"]}
                        else:
                            raise ValueError("Tool arguments must include 'actions_list' or 'action'")

                    tool_call = {"tool": action["name"], "args": env_args}

                    # Log the tool call
                    tool_call_entry = {
                        "turn": turn,
                        "type": "tool_call",
                        "tool_call": tool_call,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.debug_log.append(tool_call_entry)
                    try:
                        await tracer.record_message(
                            json.dumps(tool_call), "tool_use", metadata={"subtype": "tool_call"}
                        )
                    except Exception:
                        pass

                    # Debug: Print tool call
                    if self.verbose:
                        print(f"DEBUG: Sending tool_call: {tool_call}")

                    obs = await env.step(tool_call)

                    # Log the environment response
                    # Extract key debug fields if present
                    dbg = {
                        "agent_debug": obs.get("agent_debug"),
                        "front_cell": obs.get("front_cell"),
                        "available_actions": obs.get("available_actions"),
                        "last_action": obs.get("last_action"),
                        "last_action_result": obs.get("last_action_result"),
                    }

                    env_response_entry = {
                        "turn": turn,
                        "type": "env_response",
                        "response": obs,
                        "response_debug": dbg,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.debug_log.append(env_response_entry)
                    try:
                        await tracer.record_message(
                            json.dumps(dbg), "system", metadata={"subtype": "env_response_debug"}
                        )
                    except Exception:
                        pass

                    # Debug: Print response & grid each step
                    if self.verbose:
                        print(f"DEBUG: Environment response keys: {list(obs.keys())}")
                        if "error" in obs:
                            print(f"DEBUG: ERROR: {obs['error']}")
                        # Show formatted grid every step
                        try:
                            print(self._format_observation(obs))
                        except Exception:
                            pass

                    # Check if terminated (inside timestep); mark to break outer loop
                    if obs.get("terminated"):
                        success = obs.get("success") or "goal" in str(obs).lower()
                        if self.verbose:
                            print(
                                f"\nEpisode ended! Success: {success}, Final Reward: {total_reward:.3f}"
                            )
                        should_break = True
                # End timestep
                if should_break:
                    break

            # Track history with result
            action_reasoning = action["parameters"].get("reasoning", "")
            if "executed_actions" in obs:
                action_taken = ",".join(obs["executed_actions"])  # from multi-action env tool
            else:
                action_taken = action["parameters"].get("action", "")
            action_result = obs.get("last_action_result", "")

            # Extract position info if available
            position_info = ""
            if "observation" in obs:
                lines = obs["observation"].split("\n")
                for line in lines:
                    if "Agent Position" in line:
                        position_info = f" -> {line}"
                        break

            history_entry = f"Action(s): {action_taken} | Reasoning: {action_reasoning} | Result: {action_result}{position_info}"
            self.history.append(history_entry)

            # Update metrics
            total_reward = obs["total_reward"]
            last_reward = obs["reward_last"]

            if self.verbose:
                print(f"Reward: {last_reward:.3f} (Total: {total_reward:.3f})")
                if "observation" in obs:
                    # Just print position line for brevity
                    lines = obs["observation"].split("\n")
                    for line in lines:
                        if "Agent Position" in line:
                            print(line)
                            break

            # (termination handled inside timestep)

        # Get final metrics
        final_obs = await env.terminate()

        # Log final episode summary
        episode_summary = {
            "type": "episode_summary",
            "success": success,
            "turns": turn + 1,
            "total_reward": total_reward,
            "final_position": final_obs["final_position"],
            "total_steps": final_obs["total_steps"],
            "debug_log_entries": len(self.debug_log),
            "timestamp": datetime.now().isoformat(),
        }
        self.debug_log.append(episode_summary)

        return {
            "success": success,
            "turns": turn + 1,
            "total_reward": total_reward,
            "final_position": final_obs["final_position"],
            "total_steps": final_obs["total_steps"],
            "debug_log": self.debug_log,  # Include full debug log
        }


# --- Evaluation Function ---
async def eval_minigrid_react(
    model_name: str = "gpt-5-nano",
    num_tasks: int = 5,
    difficulty: str = "easy",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate ReAct agent on MiniGrid tasks."""
    # Generate task set
    taskset = await create_minigrid_taskset(
        num_tasks_per_difficulty={difficulty: num_tasks}, seed=42
    )

    # Initialize LLM and agent
    llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.7)
    agent = MiniGridReActAgent(llm, max_turns=15, verbose=verbose)  # Reduced max turns

    # Create debug logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = f"minigrid_debug_logs_{timestamp}"
    os.makedirs(debug_dir, exist_ok=True)

    # Run evaluation
    results = []
    all_debug_logs = []

    for i, task in enumerate(taskset.instances[:num_tasks]):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Task {i + 1}/{num_tasks}: {task.metadata.env_name}")
            print(f"{'=' * 60}")

        # Create environment
        env = MiniGridEnvironment(task)

        # Run episode
        result = await agent.run_episode(env)
        result["task_id"] = str(task.id)
        result["env_name"] = task.metadata.env_name
        result["difficulty"] = task.metadata.difficulty

        # Save debug log for this task
        debug_log = result.pop("debug_log", [])  # Remove from result to avoid duplication
        debug_log_file = os.path.join(
            debug_dir, f"task_{i + 1}_{model_name.replace('.', '_')}_debug.json"
        )
        with open(debug_log_file, "w") as f:
            json.dump(
                {
                    "task_info": {
                        "task_id": result["task_id"],
                        "env_name": result["env_name"],
                        "difficulty": result["difficulty"],
                        "model": model_name,
                    },
                    "result": result,
                    "debug_log": debug_log,
                },
                f,
                indent=2,
                default=str,
            )

        all_debug_logs.append(debug_log)
        results.append(result)

        if verbose:
            print(f"\nResult: {result}")
            print(f"Debug log saved to: {debug_log_file}")

    # Save summary debug info
    summary_debug_file = os.path.join(debug_dir, f"summary_{model_name.replace('.', '_')}.json")
    with open(summary_debug_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "timestamp": timestamp,
                "all_debug_logs": all_debug_logs,
            },
            f,
            indent=2,
            default=str,
        )

    # Compute statistics
    successes = [r["success"] for r in results]
    success_rate = sum(successes) / len(successes) if successes else 0
    avg_reward = sum(r["total_reward"] for r in results) / len(results) if results else 0
    avg_steps = sum(r["total_steps"] for r in results) / len(results) if results else 0

    summary = {
        "model": model_name,
        "num_tasks": len(results),
        "difficulty": difficulty,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "results": results,
        "debug_dir": debug_dir,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Steps: {avg_steps:.1f}")

    return summary


# --- Main ---
async def main():
    """Run a configurable MiniGrid ReAct demo (single or multiple tasks)."""
    import argparse

    ap = argparse.ArgumentParser(description="MiniGrid ReAct Agent Demo")
    ap.add_argument("--model", default="gpt-5-nano", help="Model name (default: gpt-5-nano)")
    ap.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Task difficulty (default: easy)",
    )
    ap.add_argument("--num-tasks", type=int, default=1, help="Number of tasks to run (default: 1)")
    ap.add_argument("--verbose", action="store_true", help="Print per-step observation/grid")
    args = ap.parse_args()

    print("Testing MiniGrid ReAct Agent")
    print("=" * 60)
    print(f"Model: {args.model} | Difficulty: {args.difficulty} | Tasks: {args.num_tasks}")

    summary = await eval_minigrid_react(
        model_name=args.model,
        num_tasks=args.num_tasks,
        difficulty=args.difficulty,
        verbose=args.verbose,
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Reward: {summary['avg_reward']:.3f}")
    print(f"Average Steps: {summary['avg_steps']:.1f}")
    print(f"Debug dir: {summary['debug_dir']}")


if __name__ == "__main__":
    asyncio.run(main())
