
import asyncio
import uuid
import pytest
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Deque, Set
from pydantic import BaseModel, Field
from collections import deque
from synth_ai.zyk import LM
from synth_sdk.tracing.decorators import trace_event_async
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_sdk.tracing.utils import get_system_id

# Crafter specific imports
from synth_env.examples.crafter_classic.environment import (
    CrafterClassicEnvironment,
    CrafterPublicState,
    CrafterPrivateState,
)
from synth_env.examples.crafter_classic.engine import (
    CRAFTER_ACTION_MAP,  # map of action name to int
)

# Convert CRAFTER_ACTION_MAP to ACTION_STRING_TO_INT and INT_TO_ACTION_STRING
ACTION_STRING_TO_INT: Dict[str, int] = CRAFTER_ACTION_MAP
INT_TO_ACTION_STRING: Dict[int, str] = {v: k for k, v in CRAFTER_ACTION_MAP.items()}


from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_env.tasks.core import Impetus, Intent
from synth_env.environment.tools import EnvToolCall

import logging

logging.disable(logging.CRITICAL)


# --- Helper to build crafter semantic mapping ---
def get_crafter_semantic_mapping():
    """Build the crafter semantic ID to item name mapping."""
    import crafter
    import itertools

    # Create a dummy env to get ID mappings (same as environment.py)
    dummyenv = None
    try:
        dummyenv = crafter.Env()
        max_id = (
            max(
                max(dummyenv._world._mat_ids.values()),
                max(dummyenv._sem_view._obj_ids.values()),
            )
            + 1
        )
        id_to_item = ["void"] * max_id
        for name, ind in itertools.chain(
            dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()
        ):
            if name is None:
                clean = "none"
            elif hasattr(name, "__name__"):
                clean = name.__name__
            else:
                clean = str(name)
            id_to_item[ind] = clean.lower()
        player_idx = id_to_item.index("player")
        return id_to_item, player_idx
    finally:
        if dummyenv:
            try:
                dummyenv.close()
            except Exception:
                pass
        del dummyenv


# --- Helper function to format observation for LLM ---
def format_obs_for_llm_from_states(
    pub: CrafterPublicState, priv: CrafterPrivateState
) -> str:
    inventory_str = ", ".join(f"{k}:{v}" for k, v in pub.inventory.items() if v > 0)
    if not inventory_str:
        inventory_str = "empty"

    achievements_str = ", ".join(k for k, v in pub.achievements_status.items() if v)
    if not achievements_str:
        achievements_str = "none"

    # Add map view around player using the real crafter semantic mapping
    map_view = ""
    if pub.semantic_map is not None:
        px, py = pub.player_position
        view_size = 7  # 7x7 view around player
        half_view = view_size // 2

        # Get the real crafter semantic mapping
        id_to_item, player_idx = get_crafter_semantic_mapping()

        # Create a local view around the player using same logic as _plain_grid
        map_view += f"\nLocal Map View ({view_size}x{view_size} around player):\n"
        matrix = []
        for dy in range(-half_view, half_view + 1):
            row = []
            for dx in range(-half_view, half_view + 1):
                x, y = px + dx, py + dy
                if not (
                    0 <= x < pub.semantic_map.shape[0]
                    and 0 <= y < pub.semantic_map.shape[1]
                ):
                    row.append("void")
                else:
                    idx = pub.semantic_map[x, y]
                    if dx == 0 and dy == 0:
                        row.append("player")  # Player position
                    else:
                        # Use the real crafter mapping
                        item_name = (
                            id_to_item[idx] if idx < len(id_to_item) else "unknown"
                        )
                        row.append(item_name)
            matrix.append(row)

        # Transpose the matrix like _plain_grid does
        transposed = list(zip(*matrix))
        # Convert each row to a space-separated string
        for row in transposed:
            map_view += " ".join(row) + "\n"

        # Create a legend of items actually visible in the map
        visible_items = set()
        for row in transposed:
            for item in row:
                if item not in ["void", "player"]:
                    visible_items.add(item)

        if visible_items:
            map_view += f"\nVisible items: {', '.join(sorted(visible_items))}"
        else:
            map_view += "\nNo special items visible (mostly grass/empty)"

    # Simplified observation, focusing on key elements
    return (
        f"Steps: {pub.num_steps_taken}/{pub.max_steps_episode}\n"
        f"Health: {priv.player_internal_stats.get('health', 'N/A')}\n"
        f"Inventory: {inventory_str}\n"
        f"Unlocked Achievements: {achievements_str}\n"
        f"Player Position: {pub.player_position}\n"
        f"Last Reward: {priv.reward_last_step:.2f}\n"
        f"Terminated: {priv.terminated} | Truncated: {priv.truncated}"
        f"{map_view}"
    )


# ---------------------------------- custom observation callable (Optional, can be simpler for Crafter) ------------------------------ #
# For now, let's assume the default observation from the environment is sufficient,
# or we will use the direct public/private states.
# If history is needed, we can adapt the Sokoban HistoryObservationCallable.
class CrafterHistoryObservationCallable(GetObservationCallable):
    def __init__(
        self, max_history: int = 1
    ):  # Keep only current obs for simplicity now
        self._hist_obs: Deque[str] = deque(maxlen=max_history)
        self._hist_pub_state: Deque[CrafterPublicState] = deque(maxlen=max_history)
        self._hist_priv_state: Deque[CrafterPrivateState] = deque(maxlen=max_history)

    async def get_observation(
        self, pub: CrafterPublicState, priv: CrafterPrivateState
    ) -> InternalObservation:
        if pub is None or priv is None:
            return {
                "error": "Missing public or private state in get_observation",
                "history_formatted_obs": list(self._hist_obs),
            }  # type: ignore[return-value]

        formatted_obs = format_obs_for_llm_from_states(pub, priv)
        self._hist_obs.append(formatted_obs)
        self._hist_pub_state.append(pub)
        self._hist_priv_state.append(priv)

        return {
            "public": pub,
            "private": priv,
            "formatted_obs": formatted_obs,  # Current formatted obs
            "history_formatted_obs": list(self._hist_obs),  # History of formatted obs
            "history_public_states": list(self._hist_pub_state),
            "history_private_states": list(self._hist_priv_state),
        }  # type: ignore[return-value]


# --- Pydantic Models for Tool Arguments ---
class CrafterInteractArgs(BaseModel):
    action_name: str = Field(
        description="A single action name to execute in the Crafter environment (e.g., 'move_up', 'place_stone')."
    )
    reasoning: str = Field(
        description="A brief explanation of why this action was chosen."
    )


class TerminateArgs(BaseModel):
    reason: str = Field(
        description="Reason for termination (e.g., 'all tasks complete', 'stuck', 'max_steps_reached')."
    )


# --- ReAct agent for Crafter -------------------------------------------------- #
class CrafterMove(EnvToolCall):  # Simple EnvToolCall wrapper
    def __init__(self, action: int):
        super().__init__(tool="interact", args={"action": action})


class ReActAgent:
    def __init__(self, llm, max_turns: int = 50):  # Increased max_turns for Crafter
        self.llm, self.max_turns = llm, max_turns
        self.history: List[Dict[str, Any]] = []
        self.system_name: str = "crafter-react-ex"  # Changed system name
        self.system_id: Any = get_system_id(self.system_name)
        self.system_instance_id: str = str(uuid.uuid4())
        self.last_obs_dict: Optional[Dict[str, Any]] = (
            None  # To store raw observation for terminate guardrails
        )
        self.current_achievements: Set[str] = set()  # To track unique achievements

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "crafter_interact",  # Changed tool name
                    "description": "Interacts with the Crafter environment by proposing a single action.",
                    "parameters": CrafterInteractArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "Terminates the agent's execution if the task is considered complete or no useful progress can be made.",
                    "parameters": TerminateArgs.model_json_schema(),
                },
            },
        ]

    def _format_history_for_prompt(self) -> str:
        prompt_history = []
        for entry in self.history:
            if entry["type"] == "obs":
                prompt_history.append(f"OBSERVATION:\n{entry['content']}")
            elif entry["type"] == "tool_call":
                args_str = json.dumps(entry["tool_arguments"])
                prompt_history.append(
                    f"THOUGHT:\nI will call the tool `{entry['tool_name']}` with arguments: {args_str}\nACTION: (Tool call executed)"
                )
            elif entry["type"] == "tool_response":
                prompt_history.append(
                    "TOOL_RESPONSE:\n(Action executed, new observation will follow if not terminal)"
                )
        return "\n".join(prompt_history)

    @trace_event_async(event_type="react_agent_decide")
    async def decide(
        self, obs_str: str, current_raw_obs: Dict[str, Any]
    ) -> int:  # obs_str is the formatted one
        print(f"[AGENT_DEBUG] Starting decide with obs: {obs_str[:100]}...")
        self.history.append({"type": "obs", "content": obs_str})
        self.last_obs_dict = current_raw_obs  # Store for terminate guardrail

        # Update current achievements from the raw observation
        if current_raw_obs and isinstance(
            current_raw_obs.get("public"), CrafterPublicState
        ):
            pub_state: CrafterPublicState = current_raw_obs["public"]
            for ach, unlocked in pub_state.achievements_status.items():
                if unlocked:
                    self.current_achievements.add(ach)

        formatted_prompt_history = self._format_history_for_prompt()
        print(f"[AGENT_DEBUG] History length: {len(self.history)}")

        # Updated prompt for Crafter
        prompt = (
            f"{formatted_prompt_history}\n\n"
            "Based on the history above, particularly the last observation (health, inventory, achievements, position), "
            "what is your reasoning and which tool (`crafter_interact` or `terminate`) should you call next? "
            "Prioritize actions that lead to new achievements or ensure survival (e.g., find food if health is low)."
        )

        system_message = (
            "You are an agent playing Crafter. Your goal is to survive and unlock as many achievements as possible. "
            "Review the history of observations, thoughts, and actions. "
            "Based on this history, particularly the last observation, decide on the best next action. "
            "You MUST call one of the two available tools: `crafter_interact` or `terminate`.\\n\\n"
            "Available actions for `crafter_interact` are: "
            f"{', '.join(ACTION_STRING_TO_INT.keys())}.\\n"
            "Always provide a `reasoning` field in your tool call."
        )

        print(f"[AGENT_DEBUG] Calling LLM with prompt length: {len(prompt)}")
        response_obj = await self.llm.respond_async(
            system_message=system_message, user_message=prompt, tools=self.tools
        )
        print("[AGENT_DEBUG] LLM response received")

        assert response_obj.tool_calls, "Response object didn't have tool call"
        tool_calls = None

        try:
            if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
                tool_calls = response_obj.tool_calls
                print(f"[AGENT_DEBUG] Found {len(tool_calls)} tool calls")
            # ... (rest of the try-except for tool call parsing, similar to Sokoban, but adapted for CrafterInteractArgs)

            # Simplified fallback for now, can be improved
            if not tool_calls:
                print("[AGENT_DEBUG] No tool calls found, falling back to noop")
                self.history.append(
                    {
                        "type": "tool_call",
                        "tool_name": "crafter_interact",
                        "tool_arguments": {
                            "action_name": "noop",  # Fallback action
                            "reasoning": "LLM failed to provide tool_calls, fallback to noop.",
                        },
                    }
                )
                return ACTION_STRING_TO_INT["noop"]

            if len(tool_calls) == 0:
                print("[AGENT_DEBUG] Empty tool calls list, falling back to noop")
                self.history.append(
                    {"type": "error", "content": "LLM returned empty tool_calls list."}
                )
                return ACTION_STRING_TO_INT["noop"]  # noop is action 0

            tool_call_data = tool_calls[0]
            tool_name = ""
            tool_args_str = ""

            if (
                hasattr(tool_call_data, "function")
                and hasattr(tool_call_data.function, "name")
                and hasattr(tool_call_data.function, "arguments")
            ):
                tool_name = tool_call_data.function.name
                tool_args_str = tool_call_data.function.arguments
            elif (
                isinstance(tool_call_data, dict)
                and "function" in tool_call_data
                and isinstance(tool_call_data["function"], dict)
            ):
                tool_name = tool_call_data["function"].get("name")
                tool_args_str = tool_call_data["function"].get("arguments")
                if not isinstance(
                    tool_args_str, str
                ):  # Arguments might already be a dict
                    tool_arguments_dict = tool_args_str
                    tool_args_str = json.dumps(tool_arguments_dict)
                else:  # Arguments are a JSON string
                    tool_arguments_dict = json.loads(tool_args_str)
            else:
                print(
                    "[AGENT_DEBUG] Unexpected tool_call structure, falling back to noop"
                )
                self.history.append(
                    {"type": "error", "content": "Unexpected tool_call structure."}
                )
                return ACTION_STRING_TO_INT["noop"]

            print(f"[AGENT_DEBUG] Tool name: {tool_name}, Args: {tool_args_str}")

            if not tool_args_str:  # Should not happen if structure is correct
                print(
                    f"[AGENT_DEBUG] Missing arguments for tool {tool_name}, falling back to noop"
                )
                self.history.append(
                    {
                        "type": "error",
                        "content": f"Missing arguments for tool {tool_name}. Args string: '{tool_args_str}'",
                    }
                )
                return ACTION_STRING_TO_INT["noop"]

            tool_arguments = json.loads(
                tool_args_str
            )  # tool_args_str should be a valid JSON string here

            self.history.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "tool_arguments": tool_arguments,
                }
            )

            if tool_name == "crafter_interact":
                print("[AGENT_DEBUG] Processing crafter_interact tool call")
                validated_args = CrafterInteractArgs(**tool_arguments)
                action_str = validated_args.action_name
                action_int = ACTION_STRING_TO_INT.get(action_str.lower())
                print(
                    f"[AGENT_DEBUG] Action string: {action_str}, Action int: {action_int}"
                )

                if action_int is None:
                    # Attempt to find a partial match if full match fails
                    for valid_action in ACTION_STRING_TO_INT.keys():
                        if action_str.lower() in valid_action:
                            action_int = ACTION_STRING_TO_INT[valid_action]
                            break
                    if action_int is None:
                        print(
                            f"[AGENT_DEBUG] Invalid action_name: {action_str}, falling back to noop"
                        )
                        self.history.append(
                            {
                                "type": "error",
                                "content": f"Invalid action_name: {action_str}. Falling back to noop.",
                            }
                        )
                        return ACTION_STRING_TO_INT["noop"]  # Fallback to noop
                print(f"[AGENT_DEBUG] Returning action: {action_int}")
                return action_int

            elif tool_name == "terminate":
                print("[AGENT_DEBUG] Processing terminate tool call")
                # Basic terminate, could add guardrails like in Sokoban (e.g., check steps, health)
                # For now, if LLM decides to terminate, we allow it.
                # A guardrail could be: if health is critical and not many achievements, don't terminate.
                # Or if max_steps nearly reached.
                if self.last_obs_dict and self.last_obs_dict.get("private"):
                    priv_state: CrafterPrivateState = self.last_obs_dict["private"]
                    if priv_state.terminated or priv_state.truncated:
                        # If env already terminated/truncated, agent's decision to terminate is valid.
                        print(
                            "[AGENT_DEBUG] Environment already terminated/truncated, returning -1"
                        )
                        return -1
                    # Add custom logic: e.g. if agent wants to terminate because it thinks it's "done"
                    # but it's not actually a terminal state by the environment.
                    # For now, let's allow termination if the agent calls it.
                    # Consider adding a check like: if len(self.current_achievements) < threshold etc.
                    # print(f"Agent decided to terminate. Reason: {tool_arguments.get('reason')}")

                print("[AGENT_DEBUG] Agent decided to terminate, returning -1")
                return -1  # Signal termination to the main loop

            else:
                print(
                    f"[AGENT_DEBUG] Unknown tool_name: {tool_name}, falling back to noop"
                )
                self.history.append(
                    {"type": "error", "content": f"Unknown tool_name: {tool_name}"}
                )
                return ACTION_STRING_TO_INT["noop"]  # Fallback

        except Exception as e:
            # Log the exception and the problematic response_obj content if possible
            error_content = f"Error processing LLM response: {str(e)}. Response: {str(response_obj)[:500]}"
            print(f"[AGENT_DEBUG] Exception in decide: {error_content}")
            self.history.append({"type": "error", "content": error_content})
            return ACTION_STRING_TO_INT["noop"]  # Fallback


# --- Test for a single agent run ---
@pytest.mark.asyncio
async def test_react_agent_crafter(tmp_path: Path):
    # Create a simple Crafter task instance for testing
    # For Crafter, the seed in metadata is important for reproducibility.
    # initial_engine_snapshot can be None if the engine handles reset with seed.
    task_metadata = CrafterTaskInstanceMetadata(
        difficulty="easy",
        seed=42,
        # Other metadata fields can be default or placeholders if not critical for this test
        num_trees_radius=0,  # Placeholder, actual values depend on seed and world gen
        num_cows_radius=0,  # Placeholder
        num_hostiles_radius=0,  # Placeholder
    )
    inst = CrafterTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="Survive and unlock achievements."),
        intent=Intent(
            rubric={"goal": "Unlock achievements and survive"},
            gold_trajectories=None,
            gold_state_diff={},
        ),
        metadata=task_metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,  # Engine will init with seed from metadata
    )

    hist_cb = CrafterHistoryObservationCallable(max_history=1)
    env = CrafterClassicEnvironment(inst, custom_step_obs=hist_cb)
    # env.engine.package_sokoban_env.render_mode = "raw" # Not applicable to Crafter

    llm = LM(
        model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0
    )
    agent = ReActAgent(llm, max_turns=30)  # Increased for meaningful progress
    print("[DEBUG] Created agent with max_turns=30")

    async def run_episode():
        obs_payload = await env.initialize()

        if "error" in obs_payload:
            print(f"Error during env.initialize: {obs_payload['error']}")
            return False, 0

        # Initial observation for the agent
        # The CrafterHistoryObservationCallable returns a dict with 'public', 'private', 'formatted_obs'
        current_formatted_obs = obs_payload["formatted_obs"]
        raw_obs_for_agent_decision = obs_payload  # Pass the whole payload which includes public and private states

        for turn in range(agent.max_turns):
            # print(f"Turn {turn + 1}/{agent.max_turns}")
            # print(f"Agent input obs:\n{current_formatted_obs}")

            act_idx = await agent.decide(
                current_formatted_obs, raw_obs_for_agent_decision
            )

            if act_idx == -1:  # Agent decided to terminate
                # print("Agent decided to terminate.")
                obs_payload_next = (
                    obs_payload  # No new observation if terminated by agent
                )
                break

            step_result = await env.step([[CrafterMove(act_idx)]])
            obs_payload_next = step_result

            if "error" in obs_payload_next:
                print(f"Error during env.step: {obs_payload_next['error']}")
                break

            # Update observations for the next agent decision
            current_formatted_obs = obs_payload_next["formatted_obs"]
            raw_obs_for_agent_decision = obs_payload_next

            agent.history.append(
                {"type": "tool_response", "content": "Action executed"}
            )

            obs_payload = obs_payload_next

            if (
                obs_payload_next["private"].terminated
                or obs_payload_next["private"].truncated
            ):
                # print("Environment terminated or truncated.")
                break

        # Ensure obs_payload_next is defined even if loop didn't run or agent terminated early
        if "obs_payload_next" not in locals():
            obs_payload_next = obs_payload

        if "error" in obs_payload_next:
            return False, len(agent.current_achievements)

        # Success could be defined as surviving some steps or achieving something
        # For this test, let's say it's successful if it ran and terminated/truncated by env
        final_private_state: CrafterPrivateState = obs_payload_next["private"]
        episode_successful = (
            final_private_state.terminated or final_private_state.truncated
        )
        return episode_successful, len(agent.current_achievements)

    episode_completed, num_achievements = await run_episode()

    # print(f"Episode completed: {episode_completed}, Achievements: {num_achievements}")
    # print(f"Agent history: {json.dumps(agent.history, indent=2)}")

    dataset = Dataset(
        questions=[
            TrainingQuestion(
                id="crafter_ep_test",
                intent="survive and achieve",
                criteria="completed_episode_or_achieved_something",
            )
        ],
        reward_signals=[
            RewardSignal(
                question_id="crafter_ep_test",
                run_id=agent.system_instance_id,
                system_instance_id=agent.system_instance_id,
                reward=1
                if episode_completed or num_achievements > 0
                else 0,  # Reward if completed or got any achievement
                error_message=""
                if episode_completed
                else "Episode not completed as expected.",
                metadata={
                    "agent_history": agent.history,
                    "achievements_unlocked": list(agent.current_achievements),
                    "num_achievements": num_achievements,
                },
            )
        ],
    )
    # upload(dataset=dataset) # Optional: uncomment to upload trace

    assert episode_completed or num_achievements > 0, (
        "Agent failed to complete the episode or unlock any achievement in the test."
    )


async def eval_react_crafter(
    model_name: str = "gpt-4.1-nano",
    formatting_model_name: str = "gpt-4.1-nano",
    modes: List[str] = None,
    n_instances_per_mode: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run ReAct agents on Crafter instances of different difficulties,
    and returns a list of dictionaries containing aggregated results for each mode.
    """

    if modes is None:
        modes = ["easy", "hard"]

    current_model_name_for_eval = model_name

    _temp_llm_for_names = LM(
        model_name=current_model_name_for_eval,
        formatting_model_name=formatting_model_name,
        temperature=0.0,
    )
    _temp_agent_for_names = ReActAgent(_temp_llm_for_names)
    actual_system_name = (
        _temp_agent_for_names.system_name
    )  # Still useful for logging within this func

    # ------------------------------------------------------------------ helpers
    async def run_episode_eval(
        inst: CrafterTaskInstance, agent_max_turns: int
    ) -> tuple[bool, int]:
        """Run a single agent/instance episode and return (success_status, num_unique_achievements)."""
        print(f"[DEBUG] Starting episode for instance {inst.id}")
        hist_cb = CrafterHistoryObservationCallable(max_history=1)
        env = CrafterClassicEnvironment(inst, custom_step_obs=hist_cb)

        llm_for_episode = LM(
            model_name=current_model_name_for_eval,
            formatting_model_name=formatting_model_name,
            temperature=0.0,
        )
        agent = ReActAgent(
            llm_for_episode, max_turns=30
        )  # Increased for meaningful progress
        print(
            f"[DEBUG] Created agent with max_turns=30 for model {current_model_name_for_eval}"
        )

        print("[DEBUG] Initializing environment...")
        obs_payload = await env.initialize()
        print(
            f"[DEBUG] Environment initialized. Obs keys: {list(obs_payload.keys()) if isinstance(obs_payload, dict) else type(obs_payload)}"
        )
        if "error" in obs_payload:
            print(f"[DEBUG] ERROR during env.initialize: {obs_payload['error']}")
            return False, 0

        current_formatted_obs = obs_payload["formatted_obs"]
        raw_obs_for_agent_decision = obs_payload
        print(f"[DEBUG] Initial formatted obs: {current_formatted_obs[:200]}...")

        turn_count = 0
        for turn_idx in range(agent.max_turns):
            turn_count += 1
            print(f"[DEBUG] === Turn {turn_idx + 1}/{agent.max_turns} ===")
            print(f"[DEBUG] Agent deciding on obs: {current_formatted_obs[:100]}...")

            act_idx = await agent.decide(
                current_formatted_obs, raw_obs_for_agent_decision
            )
            print(f"[DEBUG] Agent decided action: {act_idx}")

            if act_idx == -1:  # agent terminated
                print(f"[DEBUG] Agent decided to terminate after {turn_count} turns")
                break

            print(f"[DEBUG] Stepping environment with action {act_idx}")
            obs_payload_next = await env.step([[CrafterMove(act_idx)]])
            print(
                f"[DEBUG] Environment step completed. Obs keys: {list(obs_payload_next.keys()) if isinstance(obs_payload_next, dict) else type(obs_payload_next)}"
            )

            if "error" in obs_payload_next:
                print(f"[DEBUG] ERROR during env.step: {obs_payload_next['error']}")
                return False, len(
                    agent.current_achievements
                )  # Return current achievements even on error

            current_formatted_obs = obs_payload_next["formatted_obs"]
            raw_obs_for_agent_decision = obs_payload_next
            agent.history.append(
                {"type": "tool_response", "content": "Action executed"}
            )

            print(f"[DEBUG] New formatted obs: {current_formatted_obs}...")

            obs_payload = obs_payload_next
            if obs_payload["private"].terminated or obs_payload["private"].truncated:
                print(
                    f"[DEBUG] Environment terminated/truncated after {turn_count} turns"
                )
                print(
                    f"[DEBUG] Terminated: {obs_payload['private'].terminated}, Truncated: {obs_payload['private'].truncated}"
                )
                break

        print(f"[DEBUG] Episode completed after {turn_count} turns")
        final_private_state: CrafterPrivateState = obs_payload["private"]
        run_successful = (
            final_private_state.terminated or final_private_state.truncated
        ) or len(agent.current_achievements) >= 1
        num_unique_achievements = len(agent.current_achievements)
        print(
            f"[DEBUG] Episode result - successful: {run_successful}, achievements: {num_unique_achievements}"
        )
        print(f"[DEBUG] Agent achievements: {list(agent.current_achievements)}")
        print(
            f"[DEBUG] Final private state - terminated: {final_private_state.terminated}, truncated: {final_private_state.truncated}"
        )
        print(f"[DEBUG] Total reward: {final_private_state.total_reward_episode}")
        return run_successful, num_unique_achievements

    # ---------------------------------------------------------------- instance factory
    async def make_crafter_instances(
        difficulty: str, n_instances: int = 3, start_seed: int = 0
    ) -> List[CrafterTaskInstance]:
        instances = []
        for i in range(n_instances):
            current_seed = start_seed + i
            metadata = CrafterTaskInstanceMetadata(
                difficulty=difficulty,
                seed=current_seed,
                num_trees_radius=0,
                num_cows_radius=0,
                num_hostiles_radius=0,
            )
            instance = CrafterTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(
                    instructions=f"Survive and unlock achievements in a {difficulty} environment."
                ),
                intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            instances.append(instance)
        return instances

    # ---------------------------------------------------------------- evaluation
    configs = []
    for mode in modes:
        if mode == "easy":
            configs.append(("easy", n_instances_per_mode, 15))
        elif mode == "hard":
            configs.append(("hard", n_instances_per_mode, 15))

    results_for_model = []  # Stores dicts for each mode for the current model
    base_seed_for_difficulty = {"easy": 1000, "hard": 2000}

    print(
        f"Starting Crafter ReAct Agent Evaluation for Model: {current_model_name_for_eval}, System: {actual_system_name}"
    )

    all_generated_task_data = []
    print("\nGenerating task instances...")
    all_tasks_for_eval: Dict[str, List[CrafterTaskInstance]] = {}
    for label, num_agents, _ in configs:
        insts = await make_crafter_instances(
            label, n_instances=num_agents, start_seed=base_seed_for_difficulty[label]
        )
        all_tasks_for_eval[label] = insts
        for inst in insts:
            instance_dict = await inst.serialize()
            all_generated_task_data.append(instance_dict)
        print(f"Generated {len(insts)} instances for {label} difficulty.")

    dataset_dir = Path(__file__).parent.parent / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    synthetic_mix_path = dataset_dir / "synthetic_mix.json"
    with open(synthetic_mix_path, "w") as f:
        json.dump(all_generated_task_data, f, indent=2)
    print(
        f"Saved all {len(all_generated_task_data)} generated task instances to {synthetic_mix_path}"
    )

    for label, num_agents, max_episode_turns in configs:
        print(
            f"\nRunning {num_agents} agents on {label} difficulty tasks (max_turns: {max_episode_turns}) for model {current_model_name_for_eval}..."
        )
        current_difficulty_instances = all_tasks_for_eval[label]
        print(f"[DEBUG] About to run {len(current_difficulty_instances)} instances")

        import time

        start_time = time.time()
        print(
            f"[DEBUG] Starting asyncio.gather for {len(current_difficulty_instances)} episodes at {start_time}"
        )
        # results_per_episode is list of (success_status, num_unique_achievements)
        results_per_episode = await asyncio.gather(
            *(
                run_episode_eval(inst, max_episode_turns)
                for inst in current_difficulty_instances
            )
        )
        end_time = time.time()
        print(
            f"[DEBUG] Completed asyncio.gather in {end_time - start_time:.2f} seconds"
        )
        print(
            f"[DEBUG] Results for model {current_model_name_for_eval} on {label}: {results_per_episode}"
        )

        num_successful_runs = sum(
            1 for r_success, _ in results_per_episode if r_success
        )
        total_unique_achievements = sum(r_ach for _, r_ach in results_per_episode)
        avg_unique_achievements = (
            total_unique_achievements / len(results_per_episode)
            if results_per_episode
            else 0.0
        )

        # Append a dictionary for this mode's results
        results_for_model.append(
            {
                "Model": current_model_name_for_eval,
                "Difficulty": label,
                "Successful Runs": f"{num_successful_runs}/{len(current_difficulty_instances)}",
                "Avg Unique Achievements": f"{avg_unique_achievements:.2f}",
            }
        )
        print(
            f"Completed {label} for model {current_model_name_for_eval}: {num_successful_runs}/{len(current_difficulty_instances)} successful, Avg. Achievements: {avg_unique_achievements:.2f}"
        )

    return results_for_model


if __name__ == "__main__":
    # To run the test:
    # import tempfile
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     asyncio.run(test_react_agent_crafter(Path(tmpdir)))

    # Run both models in parallel with 3 rollouts each
    async def run_parallel_evaluation():
        # Define the models to test
        models_to_test = [
            {"model_name": "gpt-4.1-mini", "formatting_model_name": "gpt-4.1-mini"},
            {"model_name": "gpt-4.1", "formatting_model_name": "gpt-4.1"},
        ]

        print("Starting parallel evaluation of both models...")
        print("Each model will run 3 parallel environment rollouts")

        # eval_react_crafter now returns a List[Dict[str, Any]]
        # So, results_from_all_models will be a List[List[Dict[str, Any]]]
        results_from_all_models = await asyncio.gather(
            *[
                eval_react_crafter(
                    model_name=model_config["model_name"],
                    formatting_model_name=model_config["formatting_model_name"],
                    modes=["easy"],  # Can add "hard" or other modes here
                    n_instances_per_mode=3,  # 3 parallel rollouts per model
                )
                for model_config in models_to_test
            ]
        )

        print("\n=== PARALLEL EVALUATION COMPLETED ===")
        print("Both models have finished their rollouts.")

        # Flatten the list of lists into a single list of dictionaries for the combined table
        combined_results_data = []
        for model_result_list in results_from_all_models:
            combined_results_data.extend(model_result_list)

        print("\n--- Combined Evaluation Summary Table ---")
        from tabulate import (
            tabulate,
        )  # Ensure tabulate is imported here if not globally

        if combined_results_data:  # Check if there's data to tabulate
            # When using a list of dicts, headers="keys" tells tabulate to use dict keys.
            print(
                tabulate(
                    combined_results_data,
                    headers="keys",  # Corrected: use "keys" for list of dicts
                    tablefmt="github",
                )
            )
        else:
            print("No evaluation data to display.")

    # Run the parallel evaluation
    asyncio.run(run_parallel_evaluation())
