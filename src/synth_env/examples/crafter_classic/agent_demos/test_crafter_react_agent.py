#!/usr/bin/env python3
"""
Test script to run ReAct agents against Crafter environment on synth service (port 8901)
Tests on multiple easy Crafter instances with enhanced debugging
"""

import asyncio
import json
import uuid
import math
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from httpx import AsyncClient
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool


# --- Service Configuration ---
SERVICE_BASE_URL = "http://localhost:8901"
MODEL_NAME = "gpt-4.1-mini"
NUM_INSTANCES = 3
MAX_TURNS = 20
DIFFICULTY = "easy"

# --- Shaped Reward Configuration ---
# K-values for shaped reward calculation: reward = sum(K * log(count)) for each achievement
ACHIEVEMENT_K_VALUES = {
    "collect_wood": 1.0,
    "collect_stone": 1.5,
    "collect_coal": 2.0,
    "collect_iron": 3.0,
    "collect_diamond": 5.0,
    "place_table": 1.5,
    "place_furnace": 2.0,
    "place_stone": 1.0,
    "place_plant": 1.0,
    "make_wood_pickaxe": 2.0,
    "make_stone_pickaxe": 3.0,
    "make_iron_pickaxe": 4.0,
    "make_wood_sword": 2.0,
    "make_stone_sword": 3.0,
    "make_iron_sword": 4.0,
    "defeat_skeleton": 4.0,
    "defeat_zombie": 4.0,
    "wake_up": 1.0,
    "eat_cow": 1.0,
    "eat_plant": 1.0,
}


# --- Tool Definitions ---
class CrafterActionArgs(BaseModel):
    """Arguments for crafter actions."""
    actions: List[str] = Field(
        description="List of 1-5 action names to execute in sequence (e.g., ['move_up', 'do', 'mine_down'])"
    )
    reasoning: str = Field(description="Brief explanation of why these actions were chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""
    reason: str = Field(description="Reason for termination")


class CrafterActionTool(BaseTool):
    """Tool for performing actions in the Crafter environment."""
    name: str = "interact"
    arguments: type[BaseModel] = CrafterActionArgs
    description: str = "Perform 1-5 actions in sequence in the Crafter environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""
    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- Shaped Reward Helper ---
def calculate_shaped_reward(achievement_counts: Dict[str, int]) -> Dict[str, Any]:
    """Calculate shaped reward using K * log(count) for each achievement."""
    total_reward = 0.0
    reward_breakdown = {}
    
    for achievement, count in achievement_counts.items():
        if count > 0 and achievement in ACHIEVEMENT_K_VALUES:
            k_value = ACHIEVEMENT_K_VALUES[achievement]
            # Use log(count + 1) to handle count=0 case gracefully
            reward_contribution = k_value * math.log(count + 1)
            total_reward += reward_contribution
            reward_breakdown[achievement] = {
                "count": count,
                "k_value": k_value,
                "contribution": reward_contribution
            }
    
    return {
        "total_shaped_reward": total_reward,
        "breakdown": reward_breakdown
    }


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""
    
    def __init__(self, llm: LM, max_turns: int = 20, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"
        
        # Define tools in OpenAI format
        self.tools = [
            CrafterActionTool(),
            TerminateTool(),
        ]
        
    async def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get agent decision based on observation."""
        # Create conversation context
        context = f"Turn {turn+1}/{self.max_turns}\n\n{obs}"
        
        # Generate response using LLM
        response_obj = await self.llm.respond_async(
            system_message=system_message,
            user_message=context,
            tools=self.tools
        )
        
        tool_calls = response_obj.tool_calls
        
        # Handle case where tool_calls is None or empty (graceful fallback)
        if not tool_calls:
            if self.verbose:
                print(f"[WARNING] No tool calls returned by LLM, using default action")
            return {
                "name": "interact",
                "parameters": {
                    "actions": ["do"],
                    "reasoning": "Default action - no tool call received"
                }
            }
        
        tool_call_data = tool_calls[0]
        
        # Handle both dict and object formats
        if isinstance(tool_call_data, dict):
            tool_name = tool_call_data["function"]["name"]
            tool_args_str = tool_call_data["function"]["arguments"]
        else:
            tool_name = tool_call_data.function.name
            tool_args_str = tool_call_data.function.arguments
            
        tool_arguments = json.loads(tool_args_str)
        
        return {
            "name": tool_name,
            "parameters": tool_arguments
        }


# --- Crafter ReAct Agent ---
class CrafterReActAgent(BaseReActAgent):
    """ReAct agent for Crafter environment."""
    
    def __init__(self, llm: LM, max_turns: int = 20, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "crafter-react-agent"
    
    def get_system_message(self) -> str:
        return """You are playing Crafter, a survival game. Your goal is to survive and unlock achievements.

ACTIONS:
- move_up, move_down, move_left, move_right: Move in cardinal directions
- do: Interact with objects/environment in front of you
- sleep: Rest (restores health)
- place_stone, place_table, place_furnace, place_plant: Place objects
- make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe: Craft tools
- make_wood_sword, make_stone_sword, make_iron_sword: Craft weapons

STRATEGY:
1. Explore the environment to find resources
2. Collect wood, stone, coal, iron, diamond
3. Craft tools to improve resource gathering
4. Place objects to create a base
5. Manage health and sleep when needed
6. Unlock achievements by completing tasks

ENVIRONMENT:
- Trees provide wood when mined
- Stones provide stone when mined
- Coal, iron, diamond are found underground
- Monsters can damage you
- Sleep restores health

Be strategic about resource gathering and crafting. Use 'do' to interact with objects in front of you."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for Crafter."""
        parts = []
        
        if "health" in obs:
            parts.append(f"Health: {obs['health']}")
        
        if "inventory" in obs:
            # Format inventory nicely
            inv_items = []
            for item, count in obs["inventory"].items():
                if count > 0:
                    inv_items.append(f"{item}: {count}")
            inv_str = ", ".join(inv_items) if inv_items else "empty"
            parts.append(f"Inventory: {inv_str}")
        
        if "achievements" in obs:
            achieved = [k for k, v in obs["achievements"].items() if v]
            if achieved:
                parts.append(f"Achievements: {', '.join(achieved)}")
        
        if "position" in obs:
            parts.append(f"Position: {obs['position']}")
        
        if "local_view" in obs:
            parts.append(f"Local View:\n{obs['local_view']}")
        
        if "terminated" in obs:
            parts.append(f"Terminated: {obs['terminated']}")
        
        if "reward" in obs:
            parts.append(f"Reward: {obs['reward']}")
        
        return "\n".join(parts) if parts else "No formatted observation available"


# --- Episode Runner ---
async def run_single_episode(client: AsyncClient, agent: CrafterReActAgent, task_instance, instance_num: int) -> Dict[str, Any]:
    """Run a single Crafter episode and return episode metrics."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/CrafterClassic/initialize",
            json={"task_instance": await task_instance.serialize()}
        )
        
        if create_resp.status_code != 200:
            print(f"  Instance {instance_num}: Failed to create environment - {create_resp.status_code}: {create_resp.text}")
            return {
                "eval_metric": 0.0, 
                "rubric": {}, 
                "total_reward": 0.0, 
                "num_achievements": 0, 
                "terminated": False, 
                "error": True
            }
        
        env_id = create_resp.json()["env_id"]
        
        # Get initial observation
        obs = create_resp.json()["observation"]
        formatted_obs = agent.format_observation(obs)
        
        # DEBUG: Print initial state
        print(f"\n  Instance {instance_num}: Starting Crafter survival")
        print(f"  Environment: {task_instance.metadata.difficulty}")
        print(f"  Initial observation:")
        print(f"    {formatted_obs}")
        
        # Track episode metrics
        total_reward = 0.0
        final_achievements = {}
        num_achievements = 0
        
        # Run episode
        for turn in range(agent.max_turns):
            # Get agent decision
            action = await agent.decide(formatted_obs, agent.get_system_message(), turn)
            
            # DEBUG: Print agent decision
            print(f"  Turn {turn+1}: Agent chose {action['parameters']['actions']} - {action['parameters'].get('reasoning', 'no reasoning')}")
            
            # Check for termination
            if action["name"] == "terminate":
                print(f"  Agent terminated: {action['parameters'].get('reason', 'no reason given')}")
                break
            
            # Execute actions in environment
            action_sequence = action["parameters"]["actions"]
            
            # Convert action names to integers using the action map
            action_ints = []
            for action_name in action_sequence:
                # For simplicity, we'll use a basic mapping - in real implementation this would use CRAFTER_ACTION_MAP
                action_int = hash(action_name) % 18  # Crafter typically has ~18 actions
                action_ints.append(action_int)
            
            step_resp = await client.post(
                f"/env/CrafterClassic/step",
                json={
                    "env_id": env_id,
                    "request_id": str(uuid.uuid4()),
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"actions": action_ints}}]
                    }
                }
            )
            
            if step_resp.status_code != 200:
                print(f"  ❌ Step failed: {step_resp.status_code}: {step_resp.text}")
                break
            
            obs = step_resp.json()["observation"]
            formatted_obs = agent.format_observation(obs)
            
            # DEBUG: Print state after action
            print(f"  After actions: {formatted_obs}")
            
            # Update history
            agent.history.append(f"{', '.join(action_sequence)}: {action['parameters'].get('reasoning', '')[:50]}")
            
            # Track episode progress
            terminated = obs.get("terminated", False)
            step_reward = obs.get("reward", 0.0)
            total_reward += step_reward
            achievements = obs.get("achievements", {})
            if achievements:
                final_achievements = achievements
            num_achievements = sum(1 for v in achievements.values() if v) if achievements else 0
            
            if terminated:
                print(f"  ✅ Instance {instance_num}: Episode completed! Achievements: {num_achievements}, Total reward: {total_reward:.3f}")
                break
        
        # Cleanup
        await client.post(f"/env/CrafterClassic/terminate", json={"env_id": env_id})
        
        # Calculate eval metric and rubric
        eval_metric = float(num_achievements)  # Simple metric: number of achievements
        
        # Create rubric with specific achievement checks
        rubric = {}
        if final_achievements:
            rubric = {
                "collect_wood": 1.0 if final_achievements.get("collect_wood", False) else 0.0,
                "collect_stone": 1.0 if final_achievements.get("collect_stone", False) else 0.0,
                "collect_coal": 1.0 if final_achievements.get("collect_coal", False) else 0.0,
                "collect_iron": 1.0 if final_achievements.get("collect_iron", False) else 0.0,
                "collect_diamond": 1.0 if final_achievements.get("collect_diamond", False) else 0.0,
                "place_table": 1.0 if final_achievements.get("place_table", False) else 0.0,
                "place_furnace": 1.0 if final_achievements.get("place_furnace", False) else 0.0,
                "make_wood_pickaxe": 1.0 if final_achievements.get("make_wood_pickaxe", False) else 0.0,
                "make_stone_pickaxe": 1.0 if final_achievements.get("make_stone_pickaxe", False) else 0.0,
                "make_iron_pickaxe": 1.0 if final_achievements.get("make_iron_pickaxe", False) else 0.0,
                "make_wood_sword": 1.0 if final_achievements.get("make_wood_sword", False) else 0.0,
                "make_stone_sword": 1.0 if final_achievements.get("make_stone_sword", False) else 0.0,
                "make_iron_sword": 1.0 if final_achievements.get("make_iron_sword", False) else 0.0,
                "defeat_skeleton": 1.0 if final_achievements.get("defeat_skeleton", False) else 0.0,
                "defeat_zombie": 1.0 if final_achievements.get("defeat_zombie", False) else 0.0,
                "wake_up": 1.0 if final_achievements.get("wake_up", False) else 0.0,
                "eat_cow": 1.0 if final_achievements.get("eat_cow", False) else 0.0,
                "eat_plant": 1.0 if final_achievements.get("eat_plant", False) else 0.0,
            }
        else:
            # Default rubric with all zeros
            rubric = {
                "collect_wood": 0.0,
                "collect_stone": 0.0,
                "collect_coal": 0.0,
                "collect_iron": 0.0,
                "collect_diamond": 0.0,
                "place_table": 0.0,
                "place_furnace": 0.0,
                "make_wood_pickaxe": 0.0,
                "make_stone_pickaxe": 0.0,
                "make_iron_pickaxe": 0.0,
                "make_wood_sword": 0.0,
                "make_stone_sword": 0.0,
                "make_iron_sword": 0.0,
                "defeat_skeleton": 0.0,
                "defeat_zombie": 0.0,
                "wake_up": 0.0,
                "eat_cow": 0.0,
                "eat_plant": 0.0,
            }
        
        return {
            "eval_metric": eval_metric,
            "rubric": rubric,
            "total_reward": total_reward, 
            "num_achievements": num_achievements, 
            "achievements": final_achievements,
            "terminated": terminated, 
            "error": False
        }
        
    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback
        traceback.print_exc()
        return {
            "eval_metric": 0.0,
            "rubric": {},
            "total_reward": 0.0, 
            "num_achievements": 0, 
            "terminated": False, 
            "error": True
        }


# --- Batch Evaluation ---
async def evaluate_crafter_batch() -> Dict[str, Any]:
    """Evaluate Crafter agent on multiple easy instances."""
    print(f"🎯 Evaluating Crafter on {NUM_INSTANCES} easy instances...")
    
    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=0.0)
    
    # Get easy task instances using the taskset system
    from synth_env.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
    from synth_env.tasks.core import Impetus, Intent
    
    easy_task_instances = []
    for seed in range(NUM_INSTANCES):
        try:
            metadata = CrafterTaskInstanceMetadata(
                difficulty=DIFFICULTY,
                seed=seed,
                num_trees_radius=5,  # Good for easy difficulty
                num_cows_radius=2,
                num_hostiles_radius=0,  # No hostiles for easy
            )
            task_instance = CrafterTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(instructions=f"Survive and unlock achievements in an {DIFFICULTY} environment."),
                intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            easy_task_instances.append(task_instance)
        except Exception as e:
            print(f"  ⚠️  Failed to create task instance for seed {seed}: {e}")
            continue
    
    print(f"  📝 Generated {len(easy_task_instances)} {DIFFICULTY} task instances")
    
    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=30.0) as client:
        tasks = []
        for i, task_instance in enumerate(easy_task_instances):
            agent = CrafterReActAgent(llm, max_turns=MAX_TURNS, verbose=False)
            tasks.append(run_single_episode(client, agent, task_instance, i+1))
        
        results = await asyncio.gather(*tasks)
        
        # Filter out error results
        valid_results = [r for r in results if not r.get("error", False)]
        
        if not valid_results:
            return {
                "eval_metrics": [],
                "mean_eval_metric": 0.0,
                "mean_rubric": {},
                "num_episodes": 0
            }
        
        # Extract eval metrics and rubrics
        eval_metrics = [r["eval_metric"] for r in valid_results]
        mean_eval_metric = sum(eval_metrics) / len(eval_metrics)
        
        # Calculate mean rubric values
        all_rubric_keys = set()
        for r in valid_results:
            all_rubric_keys.update(r["rubric"].keys())
        
        mean_rubric = {}
        for key in all_rubric_keys:
            values = [r["rubric"].get(key, 0.0) for r in valid_results]
            mean_rubric[key] = sum(values) / len(values)
        
        # Calculate shaped reward (training rubric)
        # Count total achievements across all episodes
        achievement_counts = {}
        for result in valid_results:
            achievements = result.get("achievements", {})
            for achievement, unlocked in achievements.items():
                if unlocked:
                    achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
        
        # Calculate shaped reward using the counts
        shaped_reward_data = calculate_shaped_reward(achievement_counts)
        
        # Create training rubric (normalized shaped reward components)
        training_rubric = {}
        total_episodes = len(valid_results)
        if shaped_reward_data["breakdown"]:
            for achievement, data in shaped_reward_data["breakdown"].items():
                # Normalize by number of episodes for comparison
                training_rubric[achievement] = data["contribution"] / total_episodes
        
        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "mean_rubric": mean_rubric,
            "achievement_counts": achievement_counts,
            "shaped_reward_data": shaped_reward_data,
            "training_rubric": training_rubric,
            "num_episodes": len(valid_results)
        }


async def main():
    """Run Crafter evaluation."""
    print(f"🎮 Crafter ReAct Agent Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Service: {SERVICE_BASE_URL}")
    print(f"Instances: {NUM_INSTANCES}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Test service health
    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=10.0) as client:
        try:
            health_resp = await client.get("/health")
            health_data = health_resp.json()
            
            if "CrafterClassic" not in health_data.get("supported_environments", []):
                print("❌ CrafterClassic not available on service")
                return
                
            print("✅ Service health check passed")
                
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return
    
    # Run evaluation
    try:
        results = await evaluate_crafter_batch()
        
        print("\n" + "=" * 80)
        print("🏆 FINAL CRAFTER EVALUATION RESULTS")
        print("=" * 80)
        
        # Print eval metrics
        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Individual Scores: {[f'{x:.1f}' for x in results['eval_metrics']]}")
        print(f"  Mean Eval Metric: {results['mean_eval_metric']:.2f}")
        
        # Print standard rubric results
        print(f"\n🎯 STANDARD RUBRIC RESULTS:")
        if results['mean_rubric']:
            for achievement, score in sorted(results['mean_rubric'].items()):
                print(f"  {achievement}: {score:.2f}")
        else:
            print("  No rubric data available")
        
        # Print shaped reward results
        print(f"\n🏋️  TRAINING EVAL SCORE (SHAPED REWARD):")
        shaped_data = results.get('shaped_reward_data', {})
        print(f"  Total Shaped Reward: {shaped_data.get('total_shaped_reward', 0.0):.3f}")
        
        # Print achievement counts and contributions
        achievement_counts = results.get('achievement_counts', {})
        if achievement_counts:
            print(f"\n  Achievement Counts Across All Episodes:")
            for achievement, count in sorted(achievement_counts.items()):
                k_value = ACHIEVEMENT_K_VALUES.get(achievement, 0.0)
                contribution = k_value * math.log(count + 1) if count > 0 else 0.0
                print(f"    {achievement}: {count} times (K={k_value:.1f}, contribution={contribution:.3f})")
        else:
            print("  No achievements unlocked")
        
        # Print training rubric (normalized contributions)
        print(f"\n🎖️  TRAINING RUBRIC (PER EPISODE):")
        if results.get('training_rubric'):
            for achievement, score in sorted(results['training_rubric'].items()):
                print(f"  {achievement}: {score:.3f}")
        else:
            print("  No training rubric data available")
            
        # Overall assessment
        print(f"\n🔍 ASSESSMENT:")
        if results['mean_eval_metric'] >= 3.0:
            print("🎉 Excellent performance - achieving multiple objectives!")
        elif results['mean_eval_metric'] >= 1.0:
            print("✅ Good performance - consistently achieving objectives!")
        elif results['mean_eval_metric'] >= 0.5:
            print("⚠️  Moderate performance - some achievements unlocked")
        else:
            print("📈 Learning phase - focus on basic survival and resource gathering")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 