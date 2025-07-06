#!/usr/bin/env python3
"""
Focused Standardized Evaluation for Sokoban and CrafterClassic
This implements a working evaluation framework for these two key environments.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path

# ============================================================================
# Core Evaluation Framework
# ============================================================================

@dataclass
class EvalResult:
    """Standardized result format for environment evaluation"""
    env_name: str
    agent_name: str
    model_name: str
    episode_id: str
    scenario: str
    
    # Core metrics
    success: bool
    final_reward: float
    total_steps: int
    episode_length_seconds: float
    
    # Achievement tracking
    achievements: List[str]
    achievement_progression: List[Dict[str, Any]]  # Step-by-step achievement unlocks
    
    # Environment-specific metrics
    env_specific_metrics: Dict[str, Any]
    
    # Intermediate data
    step_rewards: List[float]
    
    # Error handling
    error: Optional[str] = None
    terminated_early: bool = False

class EnvironmentEvaluator:
    """Base evaluator that handles common evaluation logic"""
    
    def __init__(self, env_name: str, service_url: str = "http://localhost:8001"):
        self.env_name = env_name
        self.service_url = service_url
    
    async def run_episode(self, agent, scenario_config: Dict[str, Any]) -> EvalResult:
        """Run a single episode with comprehensive tracking"""
        episode_id = f"{self.env_name}_{scenario_config.get('name', 'default')}_{int(time.time())}"
        start_time = time.time()
        
        step_rewards = []
        achievement_progression = []
        previous_achievements = set()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1. Create task instance
                task_instance = await self.create_task_instance(scenario_config)
                
                # 2. Initialize environment
                init_response = await client.post(
                    f"{self.service_url}/env/{self.env_name}/initialize",
                    json={"initial_state": task_instance, "config": {}}
                )
                
                if init_response.status_code != 200:
                    raise Exception(f"Failed to initialize: {init_response.text}")
                
                init_data = init_response.json()
                env_id = init_data["env_id"]
                observation = init_data["observation"]
                
                # 3. Reset agent
                await agent.reset()
                
                # 4. Episode loop with detailed tracking
                total_steps = 0
                max_steps = scenario_config.get("max_steps", 100)
                
                # Track initial state
                initial_achievements = self.extract_achievements(observation)
                if initial_achievements:
                    achievement_progression.append({
                        "step": 0,
                        "new_achievements": initial_achievements,
                        "total_achievements": len(initial_achievements)
                    })
                    previous_achievements.update(initial_achievements)
                
                while total_steps < max_steps and not observation.get("terminated", False):
                    # Get available tools
                    available_tools = observation.get("tools", ["interact"])
                    
                    # Agent takes action
                    action = await agent.act(observation, available_tools, self.env_name)
                    
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
                    
                    # Track achievement progression
                    current_achievements = set(self.extract_achievements(observation))
                    new_achievements = current_achievements - previous_achievements
                    
                    if new_achievements:
                        achievement_progression.append({
                            "step": total_steps,
                            "new_achievements": list(new_achievements),
                            "total_achievements": len(current_achievements)
                        })
                        previous_achievements = current_achievements
                
                # 5. Terminate environment
                await client.post(
                    f"{self.service_url}/env/{self.env_name}/terminate",
                    json={"env_id": env_id}
                )
                
                # 6. Calculate final metrics
                episode_length = time.time() - start_time
                success = self.calculate_success(observation, scenario_config)
                final_reward = sum(step_rewards)
                all_achievements = self.extract_achievements(observation)
                env_metrics = self.get_env_specific_metrics(observation, step_rewards, total_steps)
                
                return EvalResult(
                    env_name=self.env_name,
                    agent_name=agent.name,
                    model_name=agent.model_name,
                    episode_id=episode_id,
                    scenario=scenario_config.get("name", "default"),
                    success=success,
                    final_reward=final_reward,
                    total_steps=total_steps,
                    episode_length_seconds=episode_length,
                    achievements=all_achievements,
                    achievement_progression=achievement_progression,
                    env_specific_metrics=env_metrics,
                    step_rewards=step_rewards
                )
                
        except Exception as e:
            return EvalResult(
                env_name=self.env_name,
                agent_name=getattr(agent, 'name', 'Unknown'),
                model_name=getattr(agent, 'model_name', 'Unknown'),
                episode_id=episode_id,
                scenario=scenario_config.get("name", "default"),
                success=False,
                final_reward=0.0,
                total_steps=0,
                episode_length_seconds=time.time() - start_time,
                achievements=[],
                achievement_progression=[],
                env_specific_metrics={},
                step_rewards=[],
                error=str(e),
                terminated_early=True
            )
    
    async def create_task_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create environment-specific task instance - override in subclasses"""
        return {}
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        """Extract achievements - override in subclasses"""
        return []
    
    def calculate_success(self, final_obs: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Calculate success - override in subclasses"""
        return False
    
    def get_env_specific_metrics(self, final_obs: Dict[str, Any], step_rewards: List[float], total_steps: int) -> Dict[str, Any]:
        """Get environment-specific metrics - override in subclasses"""
        return {}

# ============================================================================
# Sokoban Evaluator
# ============================================================================

class SokobanEvaluator(EnvironmentEvaluator):
    def __init__(self):
        super().__init__("Sokoban")
    
    async def create_task_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Sokoban rooms - for now use simple fixed rooms"""
        difficulty = config.get("difficulty", "easy")
        
        if difficulty == "tutorial":
            # Simple 4x4 room with 1 box
            return {
                "dim_room": [4, 4],
                "room_fixed": [
                    [1, 1, 1, 1],
                    [1, 0, 3, 1], 
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]
                ],
                "room_state": [
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                    [0, 4, 0, 0], 
                    [0, 0, 0, 0]
                ],
                "boxes_on_target": 0,
                "max_steps": config.get("max_steps", 25),
                "num_boxes": 1
            }
        elif difficulty == "easy":
            # 5x5 room with 1 box, slightly more complex
            return {
                "dim_room": [5, 5],
                "room_fixed": [
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 3, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]
                ],
                "room_state": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ],
                "boxes_on_target": 0,
                "max_steps": config.get("max_steps", 50),
                "num_boxes": 1
            }
        else:  # medium
            # 6x6 room with 2 boxes
            return {
                "dim_room": [6, 6],
                "room_fixed": [
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 3, 3, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1]
                ],
                "room_state": [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ],
                "boxes_on_target": 0,
                "max_steps": config.get("max_steps", 100),
                "num_boxes": 2
            }
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        achievements = []
        
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        steps_taken = observation.get("steps_taken", 0)
        
        # Progress achievements
        if boxes_on_target > 0:
            achievements.append(f"placed_{boxes_on_target}_boxes")
        
        if boxes_on_target >= total_boxes // 2 and total_boxes > 1:
            achievements.append("halfway_complete")
        
        # Success achievements
        if boxes_on_target == total_boxes:
            achievements.append("puzzle_solved")
            
            # Efficiency achievements
            if steps_taken <= 20:
                achievements.append("very_efficient")
            elif steps_taken <= 30:
                achievements.append("efficient_solution")
        
        # Progress achievements
        if steps_taken > 0:
            achievements.append("started_moving")
        
        if steps_taken >= 10:
            achievements.append("persistent_effort")
        
        return achievements
    
    def calculate_success(self, final_obs: Dict[str, Any], config: Dict[str, Any]) -> bool:
        return (final_obs.get("boxes_on_target", 0) == 
                final_obs.get("num_boxes", 0))
    
    def get_env_specific_metrics(self, final_obs: Dict[str, Any], step_rewards: List[float], total_steps: int) -> Dict[str, Any]:
        boxes_placed = final_obs.get("boxes_on_target", 0)
        total_boxes = final_obs.get("num_boxes", 0)
        steps_taken = final_obs.get("steps_taken", 0)
        
        return {
            "boxes_placed": boxes_placed,
            "total_boxes": total_boxes,
            "placement_efficiency": boxes_placed / max(total_boxes, 1),
            "steps_taken": steps_taken,
            "steps_per_box": steps_taken / max(boxes_placed, 1) if boxes_placed > 0 else steps_taken,
            "room_size": final_obs.get("dim_room", [0, 0]),
            "completion_rate": boxes_placed / max(total_boxes, 1),
            "movement_efficiency": boxes_placed / max(steps_taken, 1) if steps_taken > 0 else 0
        }

# ============================================================================
# CrafterClassic Evaluator  
# ============================================================================

class CrafterEvaluator(EnvironmentEvaluator):
    def __init__(self):
        super().__init__("CrafterClassic")
    
    async def create_task_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """CrafterClassic uses default initialization"""
        return {}
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        achievements = []
        
        # Direct achievements from the game
        achievements_status = observation.get("achievements_status", {})
        for achievement, unlocked in achievements_status.items():
            if unlocked:
                achievements.append(achievement)
        
        # Additional progress achievements
        health = observation.get("health", 0)
        inventory = observation.get("inventory", {})
        
        # Health-based achievements
        if health >= 5:
            achievements.append("healthy_survivor")
        if health >= 8:
            achievements.append("excellent_health")
        
        # Inventory-based achievements
        inventory_count = len([k for k, v in inventory.items() if v > 0])
        if inventory_count >= 3:
            achievements.append("good_collector")
        if inventory_count >= 5:
            achievements.append("master_collector")
        
        # Resource-specific achievements
        if inventory.get("wood", 0) >= 5:
            achievements.append("wood_gatherer")
        if inventory.get("stone", 0) >= 3:
            achievements.append("stone_collector")
        if inventory.get("coal", 0) >= 1:
            achievements.append("coal_finder")
        
        return achievements
    
    def calculate_success(self, final_obs: Dict[str, Any], config: Dict[str, Any]) -> bool:
        # Success criteria: unlock target number of achievements
        target_achievements = config.get("target_achievements", 3)
        achievements_count = sum(final_obs.get("achievements_status", {}).values())
        return achievements_count >= target_achievements
    
    def get_env_specific_metrics(self, final_obs: Dict[str, Any], step_rewards: List[float], total_steps: int) -> Dict[str, Any]:
        achievements_status = final_obs.get("achievements_status", {})
        inventory = final_obs.get("inventory", {})
        
        return {
            "achievements_unlocked": sum(achievements_status.values()),
            "total_possible_achievements": len(achievements_status),
            "achievement_rate": sum(achievements_status.values()) / max(len(achievements_status), 1),
            "final_health": final_obs.get("health", 0),
            "inventory_diversity": len([k for k, v in inventory.items() if v > 0]),
            "total_items": sum(inventory.values()),
            "wood_collected": inventory.get("wood", 0),
            "stone_collected": inventory.get("stone", 0),
            "coal_collected": inventory.get("coal", 0),
            "iron_collected": inventory.get("iron", 0),
            "avg_reward_per_step": sum(step_rewards) / max(len(step_rewards), 1),
            "exploration_efficiency": sum(achievements_status.values()) / max(total_steps, 1)
        }

# ============================================================================
# Simple ReAct Agent for Demo
# ============================================================================

class SimpleReActAgent:
    """Demo agent with environment-specific strategies"""
    
    def __init__(self, model_name: str = "demo-agent"):
        self.model_name = model_name
        self.name = "SimpleReActAgent"
        self.step_count = 0
        self.env_context = {}
    
    async def reset(self):
        self.step_count = 0
        self.env_context = {}
    
    async def act(self, observation: Dict[str, Any], available_tools: List[str], env_name: str) -> Dict[str, Any]:
        """Environment-aware action selection"""
        self.step_count += 1
        
        if env_name == "Sokoban":
            return self._sokoban_strategy(observation)
        elif env_name == "CrafterClassic":
            return self._crafter_strategy(observation)
        else:
            # Fallback
            return {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
    
    def _sokoban_strategy(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Improved Sokoban strategy: try to be smarter about movements"""
        # Simple strategy: right, down, left, up, but with some logic
        player_pos = observation.get("player_position", [0, 0])
        boxes_on_target = observation.get("boxes_on_target", 0)
        
        # If we haven't placed any boxes yet, try different directions
        if boxes_on_target == 0:
            actions = [8, 2, 4, 6]  # right, down, left, up
        else:
            # If we've placed some boxes, be more systematic
            actions = [8, 8, 2, 2, 4, 4, 6, 6]  # favor horizontal movement
        
        action = actions[self.step_count % len(actions)]
        return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}
    
    def _crafter_strategy(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Crafter strategy focused on basic survival and resource gathering"""
        health = observation.get("health", 0)
        inventory = observation.get("inventory", {})
        
        # Strategy based on current state
        if health < 5:
            # Low health - focus on survival
            actions = ["do", "do", "move_up", "move_down"]  # Try to find food/rest
        elif inventory.get("wood", 0) < 3:
            # Need wood - move around and collect
            actions = ["move_up", "move_right", "do", "move_left", "do"]
        else:
            # Have some resources - try crafting
            actions = ["do", "do", "do", "move_up", "move_right"]
        
        action = actions[self.step_count % len(actions)]
        return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}

# ============================================================================
# Evaluation Runner
# ============================================================================

class FocusedEvaluationRunner:
    """Evaluation runner for Sokoban and CrafterClassic"""
    
    def __init__(self):
        self.evaluators = {
            "Sokoban": SokobanEvaluator(),
            "CrafterClassic": CrafterEvaluator()
        }
    
    async def run_evaluation(self, scenarios: Dict[str, List[Dict[str, Any]]], agent) -> Dict[str, List[EvalResult]]:
        """Run evaluation across specified scenarios"""
        all_results = {}
        
        for env_name, env_scenarios in scenarios.items():
            if env_name not in self.evaluators:
                print(f"⚠️ Skipping {env_name} - no evaluator available")
                continue
                
            print(f"\n🚀 Evaluating {env_name}...")
            evaluator = self.evaluators[env_name]
            env_results = []
            
            for scenario in env_scenarios:
                scenario_name = scenario.get("name", "default")
                num_episodes = scenario.get("num_episodes", 3)
                
                print(f"  📋 Scenario: {scenario_name} ({num_episodes} episodes)")
                
                for episode in range(num_episodes):
                    print(f"    Episode {episode + 1}/{num_episodes}...", end=" ")
                    
                    result = await evaluator.run_episode(agent, scenario)
                    env_results.append(result)
                    
                    if result.error:
                        print(f"❌ Error: {result.error[:50]}...")
                    else:
                        status = "✅ SUCCESS" if result.success else "❌ FAILED"
                        print(f"{status} (R: {result.final_reward:.1f}, S: {result.total_steps}, A: {len(result.achievements)})")
                        
                        if result.achievements:
                            print(f"      🏆 Achievements: {', '.join(result.achievements[:3])}{'...' if len(result.achievements) > 3 else ''}")
            
            all_results[env_name] = env_results
        
        return all_results
    
    def print_summary(self, results: Dict[str, List[EvalResult]]):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("🎯 SOKOBAN & CRAFTER EVALUATION SUMMARY")
        print("="*80)
        
        total_episodes = sum(len(env_results) for env_results in results.values())
        total_successful = sum(sum(r.success for r in env_results) for env_results in results.values())
        
        print(f"📊 Overall: {total_successful}/{total_episodes} episodes successful ({total_successful/max(total_episodes,1):.1%})")
        
        for env_name, env_results in results.items():
            if not env_results:
                continue
                
            print(f"\n🎮 {env_name.upper()}:")
            
            # Filter out error results for stats
            valid_results = [r for r in env_results if not r.error]
            if not valid_results:
                print("  ❌ All episodes failed with errors")
                continue
            
            # Core statistics
            success_rate = sum(r.success for r in valid_results) / len(valid_results)
            avg_reward = sum(r.final_reward for r in valid_results) / len(valid_results)
            avg_steps = sum(r.total_steps for r in valid_results) / len(valid_results)
            avg_duration = sum(r.episode_length_seconds for r in valid_results) / len(valid_results)
            
            print(f"  📈 Success Rate: {success_rate:.1%} ({sum(r.success for r in valid_results)}/{len(valid_results)} episodes)")
            print(f"  🏆 Average Reward: {avg_reward:.2f}")
            print(f"  👣 Average Steps: {avg_steps:.1f}")
            print(f"  ⏱️  Average Duration: {avg_duration:.1f}s")
            
            # Achievement analysis
            all_achievements = []
            for r in valid_results:
                all_achievements.extend(r.achievements)
            
            if all_achievements:
                achievement_counts = {}
                for achievement in all_achievements:
                    achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
                
                print(f"  🏅 Top Achievements:")
                sorted_achievements = sorted(achievement_counts.items(), key=lambda x: x[1], reverse=True)
                for achievement, count in sorted_achievements[:5]:
                    percentage = count / len(valid_results) * 100
                    print(f"     • {achievement}: {count}/{len(valid_results)} ({percentage:.0f}%)")
            
            # Environment-specific metrics
            if valid_results and valid_results[0].env_specific_metrics:
                print(f"  📊 Environment Metrics:")
                
                # Calculate averages for numeric metrics
                metrics_summary = {}
                for result in valid_results:
                    for key, value in result.env_specific_metrics.items():
                        if isinstance(value, (int, float)):
                            if key not in metrics_summary:
                                metrics_summary[key] = []
                            metrics_summary[key].append(value)
                
                for metric, values in metrics_summary.items():
                    avg_value = sum(values) / len(values)
                    print(f"     • {metric}: {avg_value:.2f}")
            
            # Scenario breakdown
            scenarios = {}
            for result in valid_results:
                scenario = result.scenario
                if scenario not in scenarios:
                    scenarios[scenario] = []
                scenarios[scenario].append(result)
            
            if len(scenarios) > 1:
                print(f"  📋 By Scenario:")
                for scenario_name, scenario_results in scenarios.items():
                    scenario_success = sum(r.success for r in scenario_results) / len(scenario_results)
                    print(f"     • {scenario_name}: {scenario_success:.1%} success ({len(scenario_results)} episodes)")

# ============================================================================
# Demo Configuration and Main
# ============================================================================

SOKOBAN_CRAFTER_SCENARIOS = {
    "Sokoban": [
        {
            "name": "tutorial",
            "difficulty": "tutorial",
            "max_steps": 25,
            "num_episodes": 5,
            "description": "Simple 4x4 room with 1 box"
        },
        {
            "name": "easy",
            "difficulty": "easy", 
            "max_steps": 50,
            "num_episodes": 4,
            "description": "5x5 room with 1 box"
        },
        {
            "name": "medium",
            "difficulty": "medium",
            "max_steps": 100, 
            "num_episodes": 3,
            "description": "6x6 room with 2 boxes"
        }
    ],
    "CrafterClassic": [
        {
            "name": "survival",
            "target_achievements": 2,
            "max_steps": 200,
            "num_episodes": 4,
            "description": "Focus on basic survival"
        },
        {
            "name": "exploration",
            "target_achievements": 4,
            "max_steps": 400,
            "num_episodes": 3,
            "description": "Explore and gather resources"
        },
        {
            "name": "advanced",
            "target_achievements": 6,
            "max_steps": 600,
            "num_episodes": 2,
            "description": "Advanced crafting challenges"
        }
    ]
}

async def main():
    """Run focused Sokoban and CrafterClassic evaluation"""
    print("🎯 Focused Evaluation: Sokoban & CrafterClassic")
    print("=" * 60)
    
    # Check service health
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get("http://localhost:8001/health")
            if health.status_code == 200:
                supported = health.json()["supported_environments"]
                print(f"✅ Service healthy. Environments: {supported}")
                
                if "Sokoban" not in supported or "CrafterClassic" not in supported:
                    print("⚠️ Warning: Required environments not fully supported")
            else:
                print("❌ Service not responding properly")
                return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return
    
    # Initialize and run evaluation
    agent = SimpleReActAgent("demo-sokoban-crafter-agent")
    runner = FocusedEvaluationRunner()
    
    print(f"🤖 Agent: {agent.name} ({agent.model_name})")
    print(f"📋 Total scenarios: {sum(len(scenarios) for scenarios in SOKOBAN_CRAFTER_SCENARIOS.values())}")
    print(f"🎮 Total episodes: {sum(sum(s.get('num_episodes', 3) for s in scenarios) for scenarios in SOKOBAN_CRAFTER_SCENARIOS.values())}")
    
    # Run evaluation
    results = await runner.run_evaluation(SOKOBAN_CRAFTER_SCENARIOS, agent)
    
    # Print summary
    runner.print_summary(results)
    
    # Save detailed results
    Path("results").mkdir(exist_ok=True)
    
    detailed_results = {}
    for env_name, env_results in results.items():
        detailed_results[env_name] = [asdict(result) for result in env_results]
    
    with open("results/sokoban_crafter_evaluation.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to results/sokoban_crafter_evaluation.json")
    print("🎉 Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main()) 