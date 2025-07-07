#!/usr/bin/env python3
"""
Test script to run ReAct agents against NetHack environment on synth service (port 8901)
Tests on multiple easy NetHack instances with enhanced debugging
"""

import asyncio
import json
import uuid
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
MAX_TURNS = 30
DIFFICULTY = "tutorial"


# --- Tool Definitions ---
class NetHackActionArgs(BaseModel):
    """Arguments for nethack actions."""
    actions: List[str] = Field(
        description="List of 1-3 action names to execute in sequence (e.g., ['north', 'search', 'inventory'])"
    )
    reasoning: str = Field(description="Brief explanation of why these actions were chosen")


class TerminateArgs(BaseModel):
    """Arguments for termination."""
    reason: str = Field(description="Reason for termination")


class NetHackActionTool(BaseTool):
    """Tool for performing actions in the NetHack environment."""
    name: str = "interact"
    arguments: type[BaseModel] = NetHackActionArgs
    description: str = "Perform 1-3 actions in sequence in the NetHack environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""
    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""
    
    def __init__(self, llm: LM, max_turns: int = 30, verbose: bool = False):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"
        
        # Define tools in OpenAI format
        self.tools = [
            NetHackActionTool(),
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
                    "actions": ["inventory"],
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


# --- NetHack ReAct Agent ---
class NetHackReActAgent(BaseReActAgent):
    """ReAct agent for NetHack environment."""
    
    def __init__(self, llm: LM, max_turns: int = 30, verbose: bool = False):
        super().__init__(llm, max_turns, verbose)
        self.system_name = "nethack-react-agent"
    
    def get_system_message(self) -> str:
        return """You are an expert NetHack player. Your goal is to explore the dungeon, survive, and make progress.

MOVEMENT ACTIONS:
- north, south, east, west: Move in cardinal directions
- northeast, northwest, southeast, southwest: Move diagonally
- go_up, go_down: Use stairs (must be on < or > symbol)

EXPLORATION ACTIONS:
- search: Look for secret doors or traps
- open: Open doors
- close: Close doors
- look: Examine surroundings (FREE ACTION)

INVENTORY ACTIONS:
- inventory: Check your items (FREE ACTION)
- pickup: Pick up items
- drop: Drop items
- wear: Put on armor
- wield: Equip weapon
- eat: Consume food
- drink: Drink potion
- read: Read scroll

INTERACTION:
- wait: Rest for one turn
- chat: Talk to NPCs
- pay: Pay shopkeeper
- kick: Kick something

MAP SYMBOLS:
- @ = you (the player)
- . = floor
- # = wall/corridor
- + = closed door
- - = open door
- < = stairs up
- > = stairs down
- $ = gold
- % = food
- ! = potion
- ? = scroll
- / = wand
- ) = weapon
- [ = armor
- d,f = pets (dog/cat)
- Letters = monsters

STRATEGY:
1. Explore systematically to map the dungeon
2. Collect useful items and gold
3. Manage hunger by eating food
4. Fight weak monsters for experience
5. Use 'look' and 'inventory' frequently (they're free!)
6. Be cautious around unknown monsters

Remember: NetHack is complex but rewarding. Take your time and observe carefully."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for NetHack."""
        parts = []
        
        if "ascii_map" in obs:
            parts.append("ASCII Map:")
            parts.append(obs["ascii_map"])
        
        if "message" in obs and obs["message"]:
            parts.append(f"Message: {obs['message']}")
        
        if "character_stats" in obs:
            stats = obs["character_stats"]
            stat_items = []
            for key, value in stats.items():
                if key in ["HP", "level", "gold", "score", "turn"]:
                    stat_items.append(f"{key}: {value}")
            if stat_items:
                parts.append(f"Stats: {', '.join(stat_items)}")
        
        if "inventory_summary" in obs:
            parts.append(f"Inventory: {obs['inventory_summary']}")
        
        if "hunger_status" in obs and obs["hunger_status"]:
            parts.append(f"Hunger: {obs['hunger_status']}")
        
        if "terminated" in obs:
            parts.append(f"Terminated: {obs['terminated']}")
        
        if "reward" in obs:
            parts.append(f"Reward: {obs['reward']}")
        
        return "\n".join(parts) if parts else "No formatted observation available"


# --- Episode Runner ---
async def run_single_episode(client: AsyncClient, agent: NetHackReActAgent, task_instance, instance_num: int) -> Dict[str, Any]:
    """Run a single NetHack episode and return episode metrics."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/NetHack/initialize",
            json={"task_instance": await task_instance.serialize()}
        )
        
        if create_resp.status_code != 200:
            print(f"  Instance {instance_num}: Failed to create environment - {create_resp.status_code}: {create_resp.text}")
            return {"eval_metric": 0.0, "rubric": {}, "error": True}
        
        env_id = create_resp.json()["env_id"]
        
        # Get initial observation
        obs = create_resp.json()["observation"]
        formatted_obs = agent.format_observation(obs)
        
        # DEBUG: Print initial state
        print(f"\n  Instance {instance_num}: Starting NetHack adventure")
        print(f"  Character: {task_instance.metadata.character_role}")
        print(f"  Goal: Reach depth {task_instance.metadata.target_depth}")
        print(f"  Initial observation:")
        print(f"    {formatted_obs}")
        
        # Track progress
        initial_depth = 1
        max_depth_reached = initial_depth
        max_reward = 0.0
        final_stats = {}
        
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
            
            step_resp = await client.post(
                f"/env/NetHack/step",
                json={
                    "env_id": env_id,
                    "request_id": str(uuid.uuid4()),
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"actions": action_sequence}}]
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
            
            # Track progress
            if "character_stats" in obs:
                final_stats = obs["character_stats"]
                if "dungeon_level" in final_stats:
                    current_depth = final_stats["dungeon_level"]
                    max_depth_reached = max(max_depth_reached, current_depth)
            
            reward = obs.get("reward", 0.0)
            max_reward = max(max_reward, reward)
            
            # Check if episode ended
            terminated = obs.get("terminated", False)
            
            if terminated:
                print(f"  📊 Instance {instance_num}: Episode ended at depth {max_depth_reached}, reward: {max_reward:.3f}")
                break
        
        # Cleanup
        await client.post(f"/env/NetHack/terminate", json={"env_id": env_id})
        
        # Calculate eval metric and rubric
        target_depth = task_instance.metadata.target_depth
        
        # Balrog score: NetHack-style scoring system
        # Base components from standard NetHack scoring
        balrog_score = 0.0
        
        # Depth component: 50 * (depth - 1) points
        depth_component = 50 * max(0, max_depth_reached - 1)
        balrog_score += depth_component
        
        # Experience/level component: approximate from stats
        if final_stats.get("experience", 0) > 0:
            # 4 * XP approximation
            balrog_score += 4 * final_stats.get("experience", 0)
        
        # Gold component: 1 point per zorkmid above starting
        starting_gold = 100  # typical starting gold
        current_gold = final_stats.get("gold", 0)
        if current_gold > starting_gold:
            balrog_score += (current_gold - starting_gold)
        
        # Bonus for deep exploration (1000 points per level beyond 20)
        if max_depth_reached > 20:
            deep_bonus = min(10000, 1000 * (max_depth_reached - 20))
            balrog_score += deep_bonus
        
        # Reward component as bonus points
        balrog_score += max_reward
        
        # Normalize to 0-1 range for eval metric (divide by typical good score)
        # A score of 50,000+ is considered quite good
        eval_metric = min(1.0, balrog_score / 50000.0)
        
        # Create rubric with specific achievements
        rubric = {
            "reached_target_depth": 1.0 if max_depth_reached >= target_depth else 0.0,
            "depth_progress": min(1.0, max_depth_reached / target_depth),
            "gained_experience": 1.0 if final_stats.get("experience", 0) > 0 else 0.0,
            "collected_gold": 1.0 if final_stats.get("gold", 0) > starting_gold else 0.0,
            "gained_levels": 1.0 if final_stats.get("level", 1) > 1 else 0.0,
            "survived_turns": min(1.0, len(agent.history) / 20.0),  # Normalize to 20 turns
            "positive_reward": 1.0 if max_reward > 0 else 0.0,
            "balrog_score_component": min(1.0, balrog_score / 10000.0),  # Normalized score component
        }
        
        # Success determination
        success = max_depth_reached >= target_depth or max_reward > 10.0 or balrog_score > 5000
        
        if success:
            print(f"  ✅ Instance {instance_num}: SUCCESS! Depth {max_depth_reached}, Balrog score: {balrog_score:.0f}")
        else:
            print(f"  ❌ Instance {instance_num}: Partial progress - depth {max_depth_reached}/{target_depth}, Balrog score: {balrog_score:.0f}")
        
        return {
            "eval_metric": eval_metric,
            "rubric": rubric,
            "max_depth_reached": max_depth_reached,
            "target_depth": target_depth,
            "max_reward": max_reward,
            "balrog_score": balrog_score,
            "final_stats": final_stats,
            "success": success,
            "error": False
        }
        
    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback
        traceback.print_exc()
        return {"eval_metric": 0.0, "rubric": {}, "error": True}


# --- Batch Evaluation ---
async def evaluate_nethack_batch() -> Dict[str, Any]:
    """Evaluate NetHack agent on multiple easy instances."""
    print(f"🎯 Evaluating NetHack on {NUM_INSTANCES} {DIFFICULTY} instances...")
    
    llm = LM(model_name=MODEL_NAME, formatting_model_name=MODEL_NAME, temperature=0.0)
    
    # Get task instances using the taskset system
    from synth_env.examples.nethack.taskset import create_nethack_taskset
    
    taskset = await create_nethack_taskset()
    
    # Filter for the desired difficulty
    task_instances = [
        inst for inst in taskset.instances 
        if inst.metadata.difficulty == DIFFICULTY
    ][:NUM_INSTANCES]
    
    if len(task_instances) < NUM_INSTANCES:
        print(f"  ⚠️  Only found {len(task_instances)} {DIFFICULTY} instances, using all available")
    
    print(f"  📝 Using {len(task_instances)} {DIFFICULTY} task instances")
    
    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=60.0) as client:  # Longer timeout for NetHack
        tasks = []
        for i, task_instance in enumerate(task_instances):
            agent = NetHackReActAgent(llm, max_turns=MAX_TURNS, verbose=False)
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
        
        # Extract Balrog scores
        balrog_scores = [r.get("balrog_score", 0.0) for r in valid_results]
        mean_balrog_score = sum(balrog_scores) / len(balrog_scores) if balrog_scores else 0.0
        
        # Calculate mean rubric values
        all_rubric_keys = set()
        for r in valid_results:
            all_rubric_keys.update(r["rubric"].keys())
        
        mean_rubric = {}
        for key in all_rubric_keys:
            values = [r["rubric"].get(key, 0.0) for r in valid_results]
            mean_rubric[key] = sum(values) / len(values)
        
        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "balrog_scores": balrog_scores,
            "mean_balrog_score": mean_balrog_score,
            "mean_rubric": mean_rubric,
            "num_episodes": len(valid_results)
        }


async def main():
    """Run NetHack evaluation."""
    print(f"🎮 NetHack ReAct Agent Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Service: {SERVICE_BASE_URL}")
    print(f"Instances: {NUM_INSTANCES}")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Test service health
    async with AsyncClient(base_url=SERVICE_BASE_URL, timeout=10.0) as client:
        try:
            health_resp = await client.get("/health")
            health_data = health_resp.json()
            
            if "NetHack" not in health_data.get("supported_environments", []):
                print("❌ NetHack not available on service")
                return
                
            print("✅ Service health check passed")
                
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return
    
    # Run evaluation
    try:
        results = await evaluate_nethack_batch()
        
        print("\n" + "=" * 80)
        print("🏆 FINAL NETHACK EVALUATION RESULTS")
        print("=" * 80)
        
        # Print eval metrics
        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {results['num_episodes']}")
        print(f"  Individual Scores: {[f'{x:.2f}' for x in results['eval_metrics']]}")
        print(f"  Mean Eval Metric: {results['mean_eval_metric']:.2f}")
        
        # Print Balrog scores
        print(f"\n⚔️  BALROG SCORES:")
        print(f"  Individual Scores: {[f'{x:.0f}' for x in results['balrog_scores']]}")
        print(f"  Mean Balrog Score: {results['mean_balrog_score']:.0f}")
        
        # Print rubric results
        print(f"\n🎯 RUBRIC RESULTS:")
        if results['mean_rubric']:
            for achievement, score in sorted(results['mean_rubric'].items()):
                print(f"  {achievement}: {score:.2f}")
        else:
            print("  No rubric data available")
        
        # Overall assessment
        print(f"\n🔍 ASSESSMENT:")
        balrog_score = results['mean_balrog_score']
        eval_metric = results['mean_eval_metric']
        
        if eval_metric > 0.8 or balrog_score > 40000:
            print("🎉 Excellent performance - mastering the dungeon!")
        elif eval_metric > 0.6 or balrog_score > 20000:
            print("✅ Good performance - making solid progress!")
        elif eval_metric > 0.4 or balrog_score > 10000:
            print("⚠️  Moderate performance - learning the ropes")
        elif balrog_score > 5000:
            print("📈 Decent exploration - building dungeon skills")
        else:
            print("🏃 Early exploration - focus on basic survival and movement")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 