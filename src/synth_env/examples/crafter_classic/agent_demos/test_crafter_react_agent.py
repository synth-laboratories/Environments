#!/usr/bin/env python3
"""
Test script to run ReAct agents against Crafter environment on synth service (port 8901)
Tests on multiple easy Crafter instances with enhanced debugging
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
MAX_TURNS = 20
DIFFICULTY = "easy"


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
    name: str = "crafter_interact"
    arguments: type[BaseModel] = CrafterActionArgs
    description: str = "Perform 1-5 actions in sequence in the Crafter environment."


class TerminateTool(BaseTool):
    """Tool to terminate the episode."""
    name: str = "terminate"
    arguments: type[BaseModel] = TerminateArgs
    description: str = "End the episode when finished or no progress can be made."


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
                "name": "crafter_interact",
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
async def run_single_episode(client: AsyncClient, agent: CrafterReActAgent, task_instance, instance_num: int) -> bool:
    """Run a single Crafter episode and return success status."""
    try:
        # Create environment using the task instance
        create_resp = await client.post(
            f"/env/CrafterClassic/initialize",
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
        print(f"\n  Instance {instance_num}: Starting Crafter survival")
        print(f"  Environment: {task_instance.metadata.difficulty}")
        print(f"  Initial observation:")
        print(f"    {formatted_obs}")
        
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
                        "tool_calls": [{"tool": "crafter_interact", "args": {"actions": action_ints}}]
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
            
            # Check if episode ended
            terminated = obs.get("terminated", False)
            reward = obs.get("reward", 0.0)
            achievements = obs.get("achievements", {})
            num_achievements = sum(1 for v in achievements.values() if v)
            
            if terminated:
                success = num_achievements > 0 or reward > 0
                if success:
                    print(f"  ✅ Instance {instance_num}: SUCCESS! Achieved {num_achievements} achievements, reward: {reward:.3f}")
                else:
                    print(f"  ❌ Instance {instance_num}: Terminated without significant progress")
                await client.post(f"/env/CrafterClassic/terminate", json={"env_id": env_id})
                return success
        
        print(f"  ❌ Instance {instance_num}: Failed to complete in {agent.max_turns} turns")
        
        # Cleanup
        await client.post(f"/env/CrafterClassic/terminate", json={"env_id": env_id})
        return False
        
    except Exception as e:
        print(f"  Instance {instance_num}: Error - {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Batch Evaluation ---
async def evaluate_crafter_batch() -> float:
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
        success_count = sum(results)
        success_rate = success_count / len(easy_task_instances)
        
        print(f"  📊 Crafter Results: {success_count}/{len(easy_task_instances)} solved ({success_rate:.1%})")
        return success_rate


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
        success_rate = await evaluate_crafter_batch()
        
        print("\n" + "=" * 50)
        print("🏆 FINAL CRAFTER RESULTS")
        print("=" * 50)
        print(f"Success Rate: {success_rate:.1%}")
        
        if success_rate > 0.5:
            print("🎉 Excellent performance!")
        elif success_rate > 0.3:
            print("✅ Good performance!")
        elif success_rate > 0.1:
            print("⚠️  Moderate performance")
        else:
            print("❌ Poor performance - needs improvement")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 