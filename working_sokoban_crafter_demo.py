#!/usr/bin/env python3
"""
Working Demo: Standardized Evaluation Framework for Sokoban & CrafterClassic
This demonstrates the framework structure with environments using default initialization.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path

@dataclass
class EvalResult:
    """Standardized result format"""
    env_name: str
    agent_name: str
    model_name: str
    episode_id: str
    scenario: str
    success: bool
    final_reward: float
    total_steps: int
    episode_length_seconds: float
    achievements: List[str]
    achievement_progression: List[Dict[str, Any]]
    env_specific_metrics: Dict[str, Any]
    step_rewards: List[float]
    error: Optional[str] = None

class StandardizedEvaluator:
    """Evaluator with standardized interface"""
    
    def __init__(self, env_name: str, service_url: str = "http://localhost:8001"):
        self.env_name = env_name
        self.service_url = service_url
    
    async def run_episode(self, agent, scenario_config: Dict[str, Any]) -> EvalResult:
        """Run episode with comprehensive tracking"""
        episode_id = f"{self.env_name}_{scenario_config.get('name', 'default')}_{int(time.time())}"
        start_time = time.time()
        
        step_rewards = []
        achievement_progression = []
        previous_achievements = set()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Initialize with empty state (works for all environments)
                init_response = await client.post(
                    f"{self.service_url}/env/{self.env_name}/initialize",
                    json={"initial_state": {}, "config": {}}
                )
                
                if init_response.status_code != 200:
                    raise Exception(f"Failed to initialize: {init_response.text}")
                
                init_data = init_response.json()
                env_id = init_data["env_id"]
                observation = init_data["observation"]
                
                await agent.reset()
                
                # Episode loop
                total_steps = 0
                max_steps = scenario_config.get("max_steps", 50)
                
                # Track initial achievements
                initial_achievements = self.extract_achievements(observation)
                if initial_achievements:
                    achievement_progression.append({
                        "step": 0,
                        "new_achievements": initial_achievements,
                        "total_achievements": len(initial_achievements)
                    })
                    previous_achievements.update(initial_achievements)
                
                while total_steps < max_steps and not observation.get("terminated", False):
                    available_tools = observation.get("tools", ["interact"])
                    action = await agent.act(observation, available_tools, self.env_name)
                    
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
                    
                    # Track new achievements
                    current_achievements = set(self.extract_achievements(observation))
                    new_achievements = current_achievements - previous_achievements
                    
                    if new_achievements:
                        achievement_progression.append({
                            "step": total_steps,
                            "new_achievements": list(new_achievements),
                            "total_achievements": len(current_achievements)
                        })
                        previous_achievements = current_achievements
                
                # Terminate
                await client.post(
                    f"{self.service_url}/env/{self.env_name}/terminate",
                    json={"env_id": env_id}
                )
                
                # Calculate results
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
                error=str(e)
            )
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        """Extract achievements based on environment type"""
        achievements = []
        
        if self.env_name == "Sokoban":
            boxes_on_target = observation.get("boxes_on_target", 0)
            total_boxes = observation.get("num_boxes", 0)
            steps_taken = observation.get("steps_taken", 0)
            
            if steps_taken > 0:
                achievements.append("started_moving")
            if boxes_on_target > 0:
                achievements.append(f"placed_{boxes_on_target}_boxes")
            if boxes_on_target == total_boxes and total_boxes > 0:
                achievements.append("puzzle_solved")
            if steps_taken > 0 and steps_taken <= 20 and boxes_on_target == total_boxes:
                achievements.append("efficient_solution")
                
        elif self.env_name == "CrafterClassic":
            # CrafterClassic achievements
            achievements_status = observation.get("achievements_status", {})
            for achievement, unlocked in achievements_status.items():
                if unlocked:
                    achievements.append(achievement)
            
            # Additional progress achievements
            health = observation.get("health", 0)
            inventory = observation.get("inventory", {})
            
            if health >= 7:
                achievements.append("healthy_survivor")
            
            inventory_count = len([k for k, v in inventory.items() if v > 0])
            if inventory_count >= 3:
                achievements.append("good_collector")
            
            if inventory.get("wood", 0) >= 3:
                achievements.append("wood_gatherer")
        
        return achievements
    
    def calculate_success(self, final_obs: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Calculate if episode was successful"""
        if self.env_name == "Sokoban":
            boxes_on_target = final_obs.get("boxes_on_target", 0)
            total_boxes = final_obs.get("num_boxes", 0)
            return boxes_on_target == total_boxes and total_boxes > 0
        elif self.env_name == "CrafterClassic":
            target_achievements = config.get("target_achievements", 2)
            achievements_count = sum(final_obs.get("achievements_status", {}).values())
            return achievements_count >= target_achievements
        return False
    
    def get_env_specific_metrics(self, final_obs: Dict[str, Any], step_rewards: List[float], total_steps: int) -> Dict[str, Any]:
        """Extract environment-specific metrics"""
        if self.env_name == "Sokoban":
            boxes_placed = final_obs.get("boxes_on_target", 0)
            total_boxes = final_obs.get("num_boxes", 0)
            steps_taken = final_obs.get("steps_taken", 0)
            
            return {
                "boxes_placed": boxes_placed,
                "total_boxes": total_boxes,
                "completion_rate": boxes_placed / max(total_boxes, 1),
                "steps_taken": steps_taken,
                "movement_efficiency": boxes_placed / max(steps_taken, 1) if steps_taken > 0 else 0
            }
        
        elif self.env_name == "CrafterClassic":
            achievements_status = final_obs.get("achievements_status", {})
            inventory = final_obs.get("inventory", {})
            
            return {
                "achievements_unlocked": sum(achievements_status.values()),
                "achievement_rate": sum(achievements_status.values()) / max(len(achievements_status), 1),
                "final_health": final_obs.get("health", 0),
                "inventory_diversity": len([k for k, v in inventory.items() if v > 0]),
                "wood_collected": inventory.get("wood", 0),
                "stone_collected": inventory.get("stone", 0),
                "avg_reward_per_step": sum(step_rewards) / max(len(step_rewards), 1)
            }
        
        return {}

class SimpleAgent:
    """Demo agent with environment-specific strategies"""
    
    def __init__(self, model_name: str = "demo-agent"):
        self.model_name = model_name
        self.name = "SimpleAgent"
        self.step_count = 0
    
    async def reset(self):
        self.step_count = 0
    
    async def act(self, observation: Dict[str, Any], available_tools: List[str], env_name: str) -> Dict[str, Any]:
        self.step_count += 1
        
        if env_name == "Sokoban":
            # Systematic movement pattern
            actions = [8, 2, 4, 6]  # right, down, left, up
            action = actions[self.step_count % len(actions)]
            return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}
        
        elif env_name == "CrafterClassic":
            # Survival-focused strategy
            health = observation.get("health", 0)
            if health < 5:
                actions = ["do", "do", "move_up"]  # Try to find food/rest
            else:
                actions = ["move_up", "move_right", "do", "move_down", "move_left", "do"]
            
            action = actions[self.step_count % len(actions)]
            return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}
        
        # Fallback
        return {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}

class BenchmarkResults:
    """Results aggregation and analysis"""
    
    def __init__(self, results: Dict[str, List[EvalResult]]):
        self.results = results
    
    def print_comprehensive_summary(self):
        """Print detailed evaluation summary with insights"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        # Overall statistics
        total_episodes = sum(len(env_results) for env_results in self.results.values())
        valid_episodes = sum(len([r for r in env_results if not r.error]) 
                           for env_results in self.results.values())
        total_successful = sum(sum(r.success for r in env_results if not r.error) 
                             for env_results in self.results.values())
        
        print(f"📊 Overall Performance:")
        print(f"   • Total Episodes: {total_episodes}")
        print(f"   • Valid Episodes: {valid_episodes}")
        print(f"   • Success Rate: {total_successful}/{valid_episodes} ({total_successful/max(valid_episodes,1):.1%})")
        
        for env_name, env_results in self.results.items():
            valid_results = [r for r in env_results if not r.error]
            if not valid_results:
                print(f"\n🎮 {env_name}: ❌ No valid episodes")
                continue
            
            print(f"\n🎮 {env_name.upper()}:")
            
            # Core metrics
            success_rate = sum(r.success for r in valid_results) / len(valid_results)
            avg_reward = sum(r.final_reward for r in valid_results) / len(valid_results)
            avg_steps = sum(r.total_steps for r in valid_results) / len(valid_results)
            
            print(f"   📈 Success Rate: {success_rate:.1%}")
            print(f"   🏆 Average Reward: {avg_reward:.2f}")
            print(f"   👣 Average Steps: {avg_steps:.1f}")
            
            # Achievement analysis
            all_achievements = []
            for r in valid_results:
                all_achievements.extend(r.achievements)
            
            if all_achievements:
                achievement_counts = {}
                for achievement in all_achievements:
                    achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
                
                print(f"   🏅 Achievement Highlights:")
                sorted_achievements = sorted(achievement_counts.items(), key=lambda x: x[1], reverse=True)
                for achievement, count in sorted_achievements[:3]:
                    percentage = count / len(valid_results) * 100
                    print(f"      • {achievement}: {percentage:.0f}% of episodes")
            
            # Environment-specific insights
            if valid_results[0].env_specific_metrics:
                print(f"   📊 Key Metrics:")
                metrics_summary = {}
                for result in valid_results:
                    for key, value in result.env_specific_metrics.items():
                        if isinstance(value, (int, float)):
                            if key not in metrics_summary:
                                metrics_summary[key] = []
                            metrics_summary[key].append(value)
                
                for metric, values in list(metrics_summary.items())[:4]:  # Top 4 metrics
                    avg_value = sum(values) / len(values)
                    print(f"      • {metric}: {avg_value:.2f}")
            
            # Scenario breakdown
            scenarios = {}
            for result in valid_results:
                scenario = result.scenario
                if scenario not in scenarios:
                    scenarios[scenario] = []
                scenarios[scenario].append(result)
            
            if len(scenarios) > 1:
                print(f"   📋 Scenario Breakdown:")
                for scenario_name, scenario_results in scenarios.items():
                    scenario_success = sum(r.success for r in scenario_results) / len(scenario_results)
                    scenario_achievements = sum(len(r.achievements) for r in scenario_results) / len(scenario_results)
                    print(f"      • {scenario_name}: {scenario_success:.1%} success, {scenario_achievements:.1f} avg achievements")
    
    def save_detailed_report(self, filepath: str):
        """Save comprehensive evaluation report"""
        report_data = {
            "summary": {
                "total_episodes": sum(len(env_results) for env_results in self.results.values()),
                "environments_tested": list(self.results.keys()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "environment_results": {},
            "detailed_episodes": {}
        }
        
        for env_name, env_results in self.results.items():
            valid_results = [r for r in env_results if not r.error]
            
            if valid_results:
                success_rate = sum(r.success for r in valid_results) / len(valid_results)
                avg_reward = sum(r.final_reward for r in valid_results) / len(valid_results)
                
                # Aggregate achievements
                all_achievements = []
                for r in valid_results:
                    all_achievements.extend(r.achievements)
                achievement_counts = {}
                for achievement in all_achievements:
                    achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
                
                report_data["environment_results"][env_name] = {
                    "success_rate": success_rate,
                    "average_reward": avg_reward,
                    "total_episodes": len(valid_results),
                    "achievement_counts": achievement_counts
                }
            
            report_data["detailed_episodes"][env_name] = [asdict(r) for r in env_results]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to {filepath}")

# Demo scenarios
FOCUSED_SCENARIOS = {
    "Sokoban": [
        {"name": "basic", "max_steps": 30, "num_episodes": 3, "target_achievements": 1},
        {"name": "intermediate", "max_steps": 50, "num_episodes": 2, "target_achievements": 2}
    ],
    "CrafterClassic": [
        {"name": "survival", "target_achievements": 2, "max_steps": 100, "num_episodes": 3},
        {"name": "exploration", "target_achievements": 3, "max_steps": 150, "num_episodes": 2}
    ]
}

async def main():
    """Run focused evaluation demo"""
    print("🎯 Standardized Evaluation Framework Demo")
    print("Sokoban (Spatial Reasoning) & CrafterClassic (Survival/Resource Management)")
    print("=" * 75)
    
    # Check service
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get("http://localhost:8001/health")
            if health.status_code == 200:
                supported = health.json()["supported_environments"]
                print(f"✅ Service running. Available: {supported}")
            else:
                print("❌ Service not responding")
                return
    except Exception as e:
        print(f"❌ Service connection failed: {e}")
        return
    
    # Run evaluation
    agent = SimpleAgent("gemini-1.5-flash-demo")
    all_results = {}
    
    total_scenarios = sum(len(scenarios) for scenarios in FOCUSED_SCENARIOS.values())
    total_episodes = sum(sum(s.get("num_episodes", 3) for s in scenarios) 
                        for scenarios in FOCUSED_SCENARIOS.values())
    
    print(f"🤖 Agent: {agent.name} ({agent.model_name})")
    print(f"📊 Plan: {total_scenarios} scenarios, {total_episodes} total episodes")
    
    for env_name, scenarios in FOCUSED_SCENARIOS.items():
        print(f"\n🚀 Evaluating {env_name}...")
        evaluator = StandardizedEvaluator(env_name)
        env_results = []
        
        for scenario in scenarios:
            scenario_name = scenario.get("name", "default")
            num_episodes = scenario.get("num_episodes", 3)
            
            print(f"  📋 Scenario: {scenario_name} ({num_episodes} episodes)")
            
            for episode in range(num_episodes):
                print(f"    Episode {episode + 1}/{num_episodes}...", end=" ")
                
                result = await evaluator.run_episode(agent, scenario)
                env_results.append(result)
                
                if result.error:
                    print(f"❌ Error: {result.error[:30]}...")
                else:
                    status = "✅ SUCCESS" if result.success else "❌ FAILED"
                    achievements_count = len(result.achievements)
                    print(f"{status} (R:{result.final_reward:.1f}, S:{result.total_steps}, A:{achievements_count})")
                    
                    if result.achievements:
                        print(f"         🏆 {', '.join(result.achievements[:2])}{'...' if len(result.achievements) > 2 else ''}")
        
        all_results[env_name] = env_results
    
    # Generate comprehensive report
    benchmark_results = BenchmarkResults(all_results)
    benchmark_results.print_comprehensive_summary()
    benchmark_results.save_detailed_report("results/standardized_evaluation_report.json")
    
    print("\n🎉 Standardized evaluation complete!")
    print("\n💡 This demonstrates:")
    print("   • Standardized metrics across environments")
    print("   • Achievement progression tracking")
    print("   • Environment-specific analysis")
    print("   • Comprehensive reporting & aggregation")
    print("   • Easy configuration via TOML files")

if __name__ == "__main__":
    asyncio.run(main()) 