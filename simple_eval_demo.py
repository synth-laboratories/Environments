#!/usr/bin/env python3
"""
Simplified working demo of standardized evaluation framework.
This demonstrates how to run evaluations across environments with proper task instances.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import httpx

@dataclass
class EvalResult:
    """Standardized result format"""
    env_name: str
    agent_name: str
    episode_id: str
    success: bool
    final_reward: float
    total_steps: int
    episode_length_seconds: float
    achievements: List[str]
    env_specific_metrics: Dict[str, Any]
    error: Optional[str] = None

class SimpleEvaluator:
    """Simple evaluator that works with any environment"""
    
    def __init__(self, env_name: str, service_url: str = "http://localhost:8001"):
        self.env_name = env_name
        self.service_url = service_url
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        """Extract achievements from observation"""
        achievements = []
        
        if self.env_name == "Sokoban":
            boxes_on_target = observation.get("boxes_on_target", 0)
            total_boxes = observation.get("num_boxes", 0)
            
            if boxes_on_target > 0:
                achievements.append(f"placed_{boxes_on_target}_boxes")
            if boxes_on_target == total_boxes:
                achievements.append("puzzle_solved")
            if observation.get("steps_taken", 0) <= 20 and boxes_on_target == total_boxes:
                achievements.append("efficient_solution")
                
        elif self.env_name == "TicTacToe":
            winner = observation.get("winner")
            if winner == "X":
                achievements.append("player_won")
            elif winner == "draw":
                achievements.append("forced_draw")
                
        elif self.env_name == "CrafterClassic":
            achievements_status = observation.get("achievements_status", {})
            for achievement, unlocked in achievements_status.items():
                if unlocked:
                    achievements.append(achievement)
                    
        elif self.env_name == "NetHack":
            level = observation.get("dungeon_level", 1)
            if level >= 3:
                achievements.append("reached_level_3")
            if observation.get("character_stats", {}).get("hp", 0) > 0:
                achievements.append("survived")
                
        return achievements
    
    def calculate_success(self, final_obs: Dict[str, Any]) -> bool:
        """Calculate if episode was successful"""
        if self.env_name == "Sokoban":
            return (final_obs.get("boxes_on_target", 0) == 
                    final_obs.get("num_boxes", 0))
        elif self.env_name == "TicTacToe":
            return final_obs.get("winner") == "X"
        elif self.env_name == "CrafterClassic":
            achievements_count = sum(final_obs.get("achievements_status", {}).values())
            return achievements_count >= 3
        elif self.env_name == "NetHack":
            return (final_obs.get("dungeon_level", 1) >= 3 and 
                    final_obs.get("character_stats", {}).get("hp", 0) > 0)
        else:
            return not final_obs.get("terminated", True)
    
    def get_env_specific_metrics(self, final_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environment-specific metrics"""
        if self.env_name == "Sokoban":
            return {
                "boxes_placed": final_obs.get("boxes_on_target", 0),
                "total_boxes": final_obs.get("num_boxes", 0),
                "steps_taken": final_obs.get("steps_taken", 0),
                "efficiency": final_obs.get("boxes_on_target", 0) / max(final_obs.get("num_boxes", 1), 1)
            }
        elif self.env_name == "TicTacToe":
            return {
                "winner": final_obs.get("winner"),
                "moves_made": final_obs.get("moves_made", 0)
            }
        elif self.env_name == "CrafterClassic":
            return {
                "achievements_unlocked": sum(final_obs.get("achievements_status", {}).values()),
                "health": final_obs.get("health", 0),
                "inventory_items": len([k for k, v in final_obs.get("inventory", {}).items() if v > 0])
            }
        elif self.env_name == "NetHack":
            char_stats = final_obs.get("character_stats", {})
            return {
                "max_dungeon_level": final_obs.get("dungeon_level", 1),
                "final_hp": char_stats.get("hp", 0),
                "experience_level": char_stats.get("experience_level", 1)
            }
        else:
            return {}
    
    async def run_episode(self, agent, max_steps: int = 50) -> EvalResult:
        """Run a single episode"""
        episode_id = f"{self.env_name}_{int(time.time())}"
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Initialize environment (using empty config for simplicity)
                init_response = await client.post(
                    f"{self.service_url}/env/{self.env_name}/initialize",
                    json={"initial_state": {}, "config": {}}
                )
                
                if init_response.status_code != 200:
                    raise Exception(f"Failed to initialize: {init_response.text}")
                
                init_data = init_response.json()
                env_id = init_data["env_id"]
                observation = init_data["observation"]
                
                # Reset agent
                await agent.reset()
                
                # Episode loop
                total_steps = 0
                step_rewards = []
                
                while total_steps < max_steps and not observation.get("terminated", False):
                    # Get available tools
                    available_tools = observation.get("tools", ["interact"])
                    
                    # Agent takes action
                    action = await agent.act(observation, available_tools)
                    
                    # Send action to environment
                    step_response = await client.post(
                        f"{self.service_url}/env/{self.env_name}/step",
                        json={
                            "env_id": env_id,
                            "request_id": f"step_{total_steps}",
                            "action": action
                        }
                    )
                    
                    if step_response.status_code != 200:
                        raise Exception(f"Step failed: {step_response.text}")
                    
                    step_data = step_response.json()
                    observation = step_data["observation"]
                    reward = step_data.get("reward", 0.0)
                    step_rewards.append(reward)
                    
                    total_steps += 1
                
                # Terminate environment
                await client.post(
                    f"{self.service_url}/env/{self.env_name}/terminate",
                    json={"env_id": env_id}
                )
                
                # Calculate results
                episode_length = time.time() - start_time
                success = self.calculate_success(observation)
                final_reward = sum(step_rewards)
                achievements = self.extract_achievements(observation)
                env_metrics = self.get_env_specific_metrics(observation)
                
                return EvalResult(
                    env_name=self.env_name,
                    agent_name=agent.name,
                    episode_id=episode_id,
                    success=success,
                    final_reward=final_reward,
                    total_steps=total_steps,
                    episode_length_seconds=episode_length,
                    achievements=achievements,
                    env_specific_metrics=env_metrics
                )
                
        except Exception as e:
            return EvalResult(
                env_name=self.env_name,
                agent_name="SimpleAgent",
                episode_id=episode_id,
                success=False,
                final_reward=0.0,
                total_steps=0,
                episode_length_seconds=time.time() - start_time,
                achievements=[],
                env_specific_metrics={},
                error=str(e)
            )

class SimpleAgent:
    """Simple demo agent with basic strategies"""
    
    def __init__(self, model_name: str = "demo-agent"):
        self.model_name = model_name
        self.name = "SimpleAgent"
        self.step_count = 0
    
    async def reset(self):
        self.step_count = 0
    
    async def act(self, observation: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
        """Simple action strategies for different environments"""
        self.step_count += 1
        
        # Detect environment type and use appropriate strategy
        if "boxes_on_target" in observation:  # Sokoban
            return self._sokoban_action(observation)
        elif "board" in observation:  # TicTacToe
            return self._tictactoe_action(observation)
        elif "inventory" in observation:  # CrafterClassic
            return self._crafter_action(observation)
        elif "ascii_map" in observation:  # NetHack
            return self._nethack_action(observation)
        else:
            # Default action for unknown environments
            if "interact" in available_tools:
                return {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
            else:
                return {"tool_calls": [{"tool": available_tools[0], "args": {}}]}
    
    def _sokoban_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Sokoban strategy: cycle through directions"""
        actions = [8, 2, 4, 6]  # right, down, left, up
        action = actions[self.step_count % len(actions)]
        return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}
    
    def _tictactoe_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Simple TicTacToe strategy: play center, then corners, then edges"""
        board = observation.get("board", [])
        
        # Priority positions: center, corners, edges
        priorities = ["B2", "A1", "A3", "C1", "C3", "A2", "B1", "B3", "C2"]
        
        for pos in priorities:
            row = ord(pos[0]) - ord('A')
            col = int(pos[1]) - 1
            if (0 <= row < len(board) and 0 <= col < len(board[0]) and 
                board[row][col] == " "):
                return {"tool_calls": [{"tool": "interact", "args": {"action": pos}}]}
        
        # Fallback
        return {"tool_calls": [{"tool": "interact", "args": {"action": "A1"}}]}
    
    def _crafter_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Crafter strategy: try different actions"""
        actions = ["move_up", "move_down", "move_left", "move_right", "do"]
        action = actions[self.step_count % len(actions)]
        return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}
    
    def _nethack_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Simple NetHack strategy: move in directions"""
        directions = ["north", "south", "east", "west"]
        direction = directions[self.step_count % len(directions)]
        return {"tool_calls": [{"tool": "move", "args": {"direction": direction}}]}

async def run_multi_env_evaluation():
    """Run evaluation across multiple environments"""
    print("🎯 Multi-Environment Evaluation Demo")
    print("=" * 50)
    
    # Check service
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get("http://localhost:8001/health")
            if health.status_code != 200:
                print("❌ Service not responding")
                return
            supported = health.json()["supported_environments"]
            print(f"✅ Service running. Environments: {supported}")
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return
    
    # Test environments
    test_environments = ["TicTacToe", "Sokoban", "CrafterClassic", "NetHack"]
    agent = SimpleAgent()
    
    all_results = {}
    
    for env_name in test_environments:
        print(f"\n🚀 Testing {env_name}...")
        evaluator = SimpleEvaluator(env_name)
        
        # Run 3 episodes per environment
        env_results = []
        for episode in range(3):
            print(f"  Episode {episode + 1}/3...", end=" ")
            result = await evaluator.run_episode(agent, max_steps=20)
            env_results.append(result)
            
            if result.error:
                print(f"❌ Error: {result.error}")
            else:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"{status} (Reward: {result.final_reward:.1f}, Steps: {result.total_steps})")
                if result.achievements:
                    print(f"    Achievements: {', '.join(result.achievements)}")
        
        all_results[env_name] = env_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for env_name, results in all_results.items():
        if not results or all(r.error for r in results):
            print(f"\n{env_name}: ❌ All episodes failed")
            continue
            
        valid_results = [r for r in results if not r.error]
        if not valid_results:
            continue
            
        success_rate = sum(r.success for r in valid_results) / len(valid_results)
        avg_reward = sum(r.final_reward for r in valid_results) / len(valid_results)
        avg_steps = sum(r.total_steps for r in valid_results) / len(valid_results)
        
        all_achievements = []
        for r in valid_results:
            all_achievements.extend(r.achievements)
        unique_achievements = list(set(all_achievements))
        
        print(f"\n{env_name}:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Episodes: {len(valid_results)}/{len(results)}")
        if unique_achievements:
            print(f"  Achievements: {', '.join(unique_achievements)}")
        
        # Show environment-specific metrics
        if valid_results and valid_results[0].env_specific_metrics:
            print("  Environment Metrics:")
            for key, value in valid_results[0].env_specific_metrics.items():
                if isinstance(value, (int, float)):
                    avg_value = sum(r.env_specific_metrics.get(key, 0) for r in valid_results) / len(valid_results)
                    print(f"    {key}: {avg_value:.2f}")
                else:
                    print(f"    {key}: {value}")
    
    # Save results
    results_data = {
        "summary": {},
        "detailed_results": {}
    }
    
    for env_name, results in all_results.items():
        results_data["detailed_results"][env_name] = [asdict(r) for r in results]
    
    with open("results/multi_env_evaluation.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to results/multi_env_evaluation.json")
    print("🎉 Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(run_multi_env_evaluation()) 