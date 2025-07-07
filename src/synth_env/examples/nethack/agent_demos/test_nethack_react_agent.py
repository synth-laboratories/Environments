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
    name: str = "nethack_interact"
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
                "name": "nethack_interact",
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
async def run_single_episode(client: AsyncClient, agent: NetHackReActAgent, task_instance, instance_num: int) -> bool:
    """Run a single NetHack episode and return success status."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/NetHack/initialize",
            json={"task_instance": await task_instance.serialize()}
        )
        
        if create_resp.status_code != 200:
            print(f"  Instance {instance_num}: Failed to create environment - {create_resp.status_code}: {create_resp.text}")
            return False
        
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
                        "tool_calls": [{"tool": "nethack_interact", "args": {"actions": action_sequence}}]
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
            
            # Track depth progress
            if "character_stats" in obs and "dungeon_level" in obs["character_stats"]:
                current_depth = obs["character_stats"]["dungeon_level"]
                max_depth_reached = max(max_depth_reached, current_depth)
            
            # Check if episode ended
            terminated = obs.get("terminated", False)
            reward = obs.get("reward", 0.0)
            
            if terminated:
                target_depth = task_instance.metadata.target_depth
                success = max_depth_reached >= target_depth or reward > 10.0  # Either reached target or got significant reward
                if success:
                    print(f"  ✅ Instance {instance_num}: SUCCESS! Reached depth {max_depth_reached}, reward: {reward:.3f}")
                else:
                    print(f"  ❌ Instance {instance_num}: Terminated at depth {max_depth_reached} (target: {target_depth})")
                await client.post(f"/env/NetHack/terminate", json={"env_id": env_id})
                return success
        
        # Check final progress
        target_depth = task_instance.metadata.target_depth
        if max_depth_reached >= target_depth:
            print(f"  ✅ Instance {instance_num}: SUCCESS! Reached target depth {max_depth_reached}")
            await client.post(f"/env/NetHack/terminate", json={"env_id": env_id})
            return True
        else:
            print(f"  ❌ Instance {instance_num}: Failed to reach target depth {target_depth} (reached {max_depth_reached})")
        
        # Cleanup
        await client.post(f"/env/NetHack/terminate", json={"env_id": env_id})
        return False
        
    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Batch Evaluation ---
async def evaluate_nethack_batch() -> float:
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
        success_count = sum(results)
        success_rate = success_count / len(task_instances)
        
        print(f"  📊 NetHack Results: {success_count}/{len(task_instances)} solved ({success_rate:.1%})")
        return success_rate


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
        success_rate = await evaluate_nethack_batch()
        
        print("\n" + "=" * 50)
        print("🏆 FINAL NETHACK RESULTS")
        print("=" * 50)
        print(f"Success Rate: {success_rate:.1%}")
        
        if success_rate > 0.5:
            print("🎉 Excellent performance!")
        elif success_rate > 0.3:
            print("✅ Good performance!")
        elif success_rate > 0.1:
            print("⚠️  Moderate performance")
        else:
            print("❌ Poor performance - NetHack is challenging!")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 