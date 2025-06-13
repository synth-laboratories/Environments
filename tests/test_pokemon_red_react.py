#!/usr/bin/env python3

import asyncio
import uuid
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synth_ai.zyk import LM
from src.examples.red.environment import (
    PokemonRedEnvironment,
    PokemonRedPublicState,
    PokemonRedPrivateState,
)
from src.examples.red.taskset import PokemonRedTaskInstance
from src.examples.red.agent_demos.test_synth_react import (
    ReActAgent,
    PokemonRedHistoryObservationCallable,
    PressButtonCall,
)
from src.tasks.core import Impetus, Intent, TaskInstanceMetadata


async def test_pokemon_red_react_20_steps():
    """Test Pokemon Red ReAct agent for exactly 20 steps and dump final state."""

    # Create task instance
    task_metadata = TaskInstanceMetadata()

    inst = PokemonRedTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="Start your Pokemon journey and collect badges."),
        intent=Intent(
            rubric={"goal": "Collect badges and progress"},
            gold_trajectories=None,
            gold_state_diff={},
        ),
        metadata=task_metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )

    # Setup environment and agent
    hist_cb = PokemonRedHistoryObservationCallable(
        max_history=3
    )  # Keep more history for inspection
    env = PokemonRedEnvironment(inst, custom_step_obs=hist_cb)

    llm = LM(
        model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0
    )
    agent = ReActAgent(llm, max_turns=20)

    print("🎮 Starting Pokemon Red ReAct Agent Test (20 steps)")
    print("=" * 60)

    # Initialize environment
    obs_payload = await env.initialize()
    if "error" in obs_payload:
        print(f"❌ Error during env.initialize: {obs_payload['error']}")
        return

    current_formatted_obs = obs_payload["formatted_obs"]
    raw_obs_for_agent_decision = obs_payload

    print("📊 Initial Observation:")
    print(current_formatted_obs)
    print("\n" + "=" * 60)

    # Track rewards
    step_rewards = []
    total_reward = 0

    # Run exactly 20 steps
    for step in range(20):
        print(f"\n🎯 Step {step + 1}/20")
        print("-" * 40)

        # Agent decides on action
        button = await agent.decide(current_formatted_obs, raw_obs_for_agent_decision)

        if button == "TERMINATE":
            print("🛑 Agent decided to terminate early")
            break

        print(f"🔘 Agent chose button: {button}")

        # Get the reasoning from the last tool call
        if agent.history and agent.history[-1]["type"] == "tool_call":
            reasoning = agent.history[-1]["tool_arguments"].get(
                "reasoning", "No reasoning provided"
            )
            print(f"💭 Reasoning: {reasoning}")

        # Execute action
        step_result = await env.step([[PressButtonCall(button)]])
        obs_payload_next = step_result

        if "error" in obs_payload_next:
            print(f"❌ Error during env.step: {obs_payload_next['error']}")
            break

        # Track rewards
        step_reward = obs_payload_next["private"].reward_last_step
        step_rewards.append(step_reward)
        total_reward += step_reward

        print(f"🏆 Step reward: {step_reward:.3f}")
        print(f"📈 Total reward: {total_reward:.3f}")

        # Update observations
        current_formatted_obs = obs_payload_next["formatted_obs"]
        raw_obs_for_agent_decision = obs_payload_next

        agent.history.append({"type": "tool_response", "content": "Button pressed"})

        obs_payload = obs_payload_next

        # Check if environment terminated
        if (
            obs_payload_next["private"].terminated
            or obs_payload_next["private"].truncated
        ):
            print("🏁 Environment terminated/truncated")
            break

    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)

    # Final state inspection
    final_public: PokemonRedPublicState = obs_payload["public"]
    final_private: PokemonRedPrivateState = obs_payload["private"]

    print("🎮 Final Game State:")
    print(f"  • HP: {final_public.party_hp_current}/{final_public.party_hp_max}")
    print(f"  • Badges: {final_public.badges}/8")
    print(f"  • Position: ({final_public.player_x}, {final_public.player_y})")
    print(f"  • Map ID: {final_public.map_id}")
    print(f"  • Party Level: {final_public.party_level}")
    print(f"  • Steps taken: {final_public.step_count}")
    print(f"  • In Battle: {final_public.in_battle}")
    print(f"  • Terminated: {final_private.terminated}")
    print(f"  • Truncated: {final_private.truncated}")

    print("\n🏆 Reward Summary:")
    print(f"  • Total reward: {total_reward:.3f}")
    print(
        f"  • Average step reward: {(total_reward / len(step_rewards)):.3f}"
        if step_rewards
        else "N/A"
    )
    print(f"  • Step rewards: {step_rewards}")

    print("\n🧠 Agent Performance:")
    print(f"  • Badges collected: {agent.current_badges}")
    print(f"  • Steps completed: {len(step_rewards)}")
    print(f"  • Final observation:\n{current_formatted_obs}")

    # Show the final LLM prompt that would be generated
    print("\n💬 Final Agent History (last 3 entries):")
    for i, entry in enumerate(agent.history[-3:]):
        print(
            f"  {i + 1}. {entry['type']}: {str(entry.get('content', entry.get('tool_arguments', '')))[:100]}..."
        )

    print("\n📝 Next LLM Prompt Preview:")
    print("-" * 40)
    formatted_history = agent._format_history_for_prompt()
    next_prompt = (
        f"{formatted_history}\n\n"
        "Based on the history above, particularly the last observation (HP, badges, position), "
        "what is your reasoning and which tool (`pokemon_red_interact` or `terminate`) should you call next? "
        "Focus on making progress: collect badges, heal when HP is low, explore new areas."
    )
    print(next_prompt)


if __name__ == "__main__":
    asyncio.run(test_pokemon_red_react_20_steps())
