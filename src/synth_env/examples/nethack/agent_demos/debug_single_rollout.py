#!/usr/bin/env python3
"""
Debug script for NetHack single rollout.
"""

import asyncio
import json
import uuid
import traceback
from pprint import pprint

from src.synth_env.examples.nethack.environment import NetHackEnvironment
from src.synth_env.examples.nethack.taskset import NetHackTaskInstance, NetHackTaskInstanceMetadata
from src.synth_env.examples.nethack.agent_demos.test_synth_react import NetHackReActAgent
from src.synth_env.examples.nethack.achievements import NetHackAchievements
from src.synth_env.examples.nethack.engine import NetHackObservationCallable
from src.synth_env.tasks.core import Impetus, Intent
from synth_ai.zyk import LM

async def debug_single_rollout():
    """Run a single NetHack rollout with extensive debugging."""
    
    print("🔍 Starting NetHack debug rollout...")
    
    try:
        # Create task instance
        print("📋 Creating task instance...")
        metadata = NetHackTaskInstanceMetadata(
            character_role="knight",
            starting_level=1,
            target_depth=5,
            time_limit=100,
            difficulty="easy",
            special_objectives=["Survive for as long as possible", "Collect gold", "Kill monsters"],
            seed=1000
        )
        
        instance = NetHackTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(instructions="Explore the NetHack dungeon on easy difficulty. Survive as long as possible, kill monsters, collect items, and descend to deeper levels."),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        print("✅ Task instance created successfully")
        
        # Setup environment
        print("🌍 Setting up environment...")
        obs_callback = NetHackObservationCallable()
        env = NetHackEnvironment(instance, custom_step_obs=obs_callback)
        print("✅ Environment created successfully")
        
        # Setup agent
        print("🤖 Setting up agent...")
        llm = LM(model_name="o3", formatting_model_name="o3", temperature=0.0)
        agent = NetHackReActAgent(llm, max_turns=10)
        
        # Set system prompt for agent
        task_instructions = instance.impetus.instructions
        agent.system_prompt = agent._create_system_prompt(task_instructions)
        print("✅ Agent created successfully")
        print(f"📝 System prompt preview: {agent.system_prompt[:200]}...")
        
        # Initialize environment
        print("\n🚀 Initializing environment...")
        obs_payload = await env.initialize()
        print("✅ Environment initialized")
        
        # Initialize reward tracking
        total_balrog_reward = 0.0
        total_standard_reward = 0.0
        reward_history = []
        achievements_unlocked = set()
        
        print("\n📊 Initial observation structure:")
        print(f"  Keys: {list(obs_payload.keys())}")
        for key, value in obs_payload.items():
            if key == "message":
                print(f"  {key}: '{value}'")
            elif key == "player_stats" and isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            elif key == "ascii_map":
                print(f"  {key}: [map with {len(str(value))} chars]")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Run a few turns
        for turn in range(1, 8):  # More turns to see o3's learning
            print(f"\n🔄 === TURN {turn} ===")
            
            # Format observation using agent's method (which includes recent actions)
            current_formatted_obs = agent._format_observation(obs_payload)
            print("📝 Using agent's format_observation method")
            
            print(f"📝 Observation preview (first 300 chars):")
            print(f"   {current_formatted_obs[:300]}...")
            
            # Agent decision
            print("🤖 Getting agent decision...")
            try:
                decision = await agent.decide(current_formatted_obs)
                print(f"✅ Agent decision received: {type(decision)}")
                
                if isinstance(decision, dict):
                    print(f"   Decision keys: {list(decision.keys())}")
                    for key, value in decision.items():
                        if key == "actions" and isinstance(value, list):
                            print(f"   {key}: {value[:3]}..." if len(value) > 3 else f"   {key}: {value}")
                        else:
                            print(f"   {key}: {type(value).__name__} ({str(value)[:50]}...)" if len(str(value)) > 50 else f"   {key}: {value}")
                else:
                    print(f"   Decision: {decision}")
                
            except Exception as e:
                print(f"❌ Agent decision failed: {e}")
                print(f"   Error type: {type(e).__name__}")
                traceback.print_exc()
                break
            
            # Extract actions
            print("⚙️ Extracting actions...")
            try:
                if isinstance(decision, dict):
                    # Handle tool call format: {'name': 'tool_name', 'parameters': {...}}
                    if decision.get("name") == "terminate":
                        print("🛑 Agent requested termination")
                        break
                    
                    # Extract actions from parameters
                    if "parameters" in decision and isinstance(decision["parameters"], dict):
                        params = decision["parameters"]
                        print(f"   Parameters: {params}")
                        if "actions" in params:
                            actions = params["actions"]
                        elif "action" in params:
                            actions = [params["action"]]
                        else:
                            actions = ["wait"]
                    elif "actions" in decision:
                        actions = decision["actions"]
                    elif "action" in decision:
                        actions = [decision["action"]]
                    else:
                        actions = ["wait"]
                else:
                    if decision == -1 or decision == [-1]:
                        print("🛑 Agent returned termination signal")
                        break
                    elif isinstance(decision, list):
                        actions = decision
                    else:
                        actions = [str(decision)]
                
                if not isinstance(actions, list):
                    actions = [str(actions)]
                
                print(f"✅ Actions extracted: {actions}")
                
            except Exception as e:
                print(f"❌ Action extraction failed: {e}")
                traceback.print_exc()
                actions = ["wait"]
            
            # Execute actions
            print(f"🎮 Executing {len(actions)} actions...")
            for i, action in enumerate(actions):
                print(f"   Action {i+1}/{len(actions)}: '{action}'")
                try:
                    obs_payload = await env.step(action)
                    print(f"   ✅ Action executed successfully")
                    
                    # Update the recent action result with the message from this action
                    result_msg = obs_payload.get("message", "").strip()
                    if not result_msg:
                        result_msg = "No message"
                    agent._update_last_action_result(result_msg)
                    
                    # Track rewards
                    balrog_reward = obs_payload.get("balrog_reward_last", 0.0)
                    standard_reward = obs_payload.get("reward_last", 0.0)
                    
                    if balrog_reward > 0:
                        total_balrog_reward += balrog_reward
                        reward_history.append({
                            "turn": turn,
                            "action": action,
                            "balrog_reward": balrog_reward,
                            "standard_reward": standard_reward,
                            "message": result_msg
                        })
                        print(f"   🏆 BALROG REWARD: +{balrog_reward:.3f} (total: {total_balrog_reward:.3f})")
                    
                    if standard_reward != 0:
                        total_standard_reward += standard_reward
                        if balrog_reward == 0:  # Don't double-print if we already showed BALROG reward
                            print(f"   ⭐ STANDARD REWARD: {standard_reward:+.3f} (total: {total_standard_reward:.3f})")
                    
                    # Track achievements
                    if "achievements_unlocked" in obs_payload:
                        current_achievements = {k for k, v in obs_payload["achievements_unlocked"].items() if v}
                        new_achievements = current_achievements - achievements_unlocked
                        if new_achievements:
                            achievements_unlocked.update(new_achievements)
                            print(f"   🎉 NEW ACHIEVEMENTS: {', '.join(new_achievements)}")
                            print(f"   📊 Total achievements: {len(achievements_unlocked)}")
                    
                    # Check for REAL environment errors (not NetHack game messages)
                    if "error" in obs_payload:
                        error_msg = obs_payload["error"]
                        # NetHack game messages like "No stairs here" are normal, not environment errors
                        if error_msg and not any(phrase in error_msg.lower() for phrase in [
                            "no stairs", "can't go", "there is nothing", "you can't", 
                            "you don't", "you aren't", "you have no", "invalid action",
                            "stairs here to", "can't", "there's nothing", "no door",
                            "nothing here", "you see nothing", "you cannot"
                        ]):
                            print(f"   ⚠️ Real environment error: {error_msg}")
                            break
                        else:
                            print(f"   ℹ️ NetHack game message: {error_msg}")
                            # This is just normal game feedback, continue playing
                    
                    # Check termination
                    private_state = obs_payload.get("private")
                    if private_state:
                        terminated = getattr(private_state, "terminated", False)
                        truncated = getattr(private_state, "truncated", False)
                        if terminated or truncated:
                            reason = "timeout" if truncated else "death"
                            print(f"   🛑 Game ended: {reason}")
                            return
                
                except Exception as e:
                    print(f"   ❌ Action execution failed: {e}")
                    traceback.print_exc()
                    return
            
            # Show updated stats
            player_stats = obs_payload.get('player_stats', {})
            depth = player_stats.get('depth', 1)
            level = player_stats.get('experience_level', 1)
            gold = player_stats.get('gold', 0)
            hp = player_stats.get('hp', 0)
            max_hp = player_stats.get('max_hp', 0)
            
            print(f"📊 Current stats: Depth={depth}, Level={level}, Gold={gold}, HP={hp}/{max_hp}")
            
            # Check for new message
            message = obs_payload.get('message', '')
            if message:
                print(f"💬 Message: '{message}'")
        
        print("\n✅ Debug rollout completed successfully!")
        
        # Print final reward summary
        print("\n" + "="*60)
        print("🏆 FINAL REWARD SUMMARY")
        print("="*60)
        print(f"💰 Total BALROG Reward: {total_balrog_reward:.3f}")
        print(f"⭐ Total Standard Reward: {total_standard_reward:.3f}")
        print(f"🎯 Total Achievements: {len(achievements_unlocked)}")
        
        if achievements_unlocked:
            print(f"🏅 Achievements Unlocked: {', '.join(sorted(achievements_unlocked))}")
        
        if reward_history:
            print(f"\n📋 REWARD BREAKDOWN ({len(reward_history)} reward events):")
            for i, reward_event in enumerate(reward_history, 1):
                print(f"  {i}. Turn {reward_event['turn']}: {reward_event['action']} → +{reward_event['balrog_reward']:.3f} BALROG")
                if reward_event['message']:
                    print(f"     Message: {reward_event['message']}")
        
        # Calculate final scores
        final_obs = obs_payload
        final_balrog_total = final_obs.get("balrog_total_reward", total_balrog_reward)
        final_score = final_obs.get("score", 0)
        final_level = final_obs.get("experience_level", 1)
        final_depth = final_obs.get("dungeon_level", 1)
        
        print(f"\n📊 FINAL GAME STATE:")
        print(f"   BALROG Total: {final_balrog_total:.3f}")
        print(f"   NetHack Score: {final_score}")
        print(f"   Experience Level: {final_level}")
        print(f"   Dungeon Depth: {final_depth}")
        
        # BALROG score is already a percentage (0-100) based on dungeon/level progression
        balrog_percentage = final_obs.get("achievement_stats", {}).get("balrog_score", 0.0)
        print(f"   BALROG Score: {balrog_percentage:.1f}% (dungeon/level progression)")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Debug rollout failed: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_single_rollout()) 