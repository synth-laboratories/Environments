#!/usr/bin/env python3
"""
Focused Sokoban Evaluation with Proper Room Generation
This uses the working room generation approach from the test suite.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path

@dataclass
class SokobanResult:
    """Sokoban-specific evaluation result"""
    episode_id: str
    scenario: str
    agent_name: str
    model_name: str
    
    # Success metrics
    success: bool
    boxes_placed: int
    total_boxes: int
    completion_rate: float
    
    # Efficiency metrics
    steps_taken: int
    max_steps: int
    step_efficiency: float
    movement_efficiency: float
    
    # Game state
    final_reward: float
    episode_length_seconds: float
    room_size: List[int]
    
    # Achievement tracking
    achievements: List[str]
    achievement_progression: List[Dict[str, Any]]
    
    # Step-by-step data
    step_rewards: List[float]
    step_observations: List[Dict[str, Any]]
    
    # Error handling
    error: Optional[str] = None
    terminated_early: bool = False

class SokobanEvaluator:
    """Focused Sokoban evaluator with proper room generation"""
    
    def __init__(self, service_url: str = "http://localhost:8001"):
        self.service_url = service_url
    
    def create_room_snapshot(self, difficulty: str, seed: int = 42) -> Dict[str, Any]:
        """Create a valid Sokoban room using the working approach"""
        from synth_env.examples.sokoban.engine_helpers.room_utils import generate_room
        
        # Configure room based on difficulty
        if difficulty == "tutorial":
            dim = (4, 4)
            num_boxes = 1
            search_depth = 10
            max_steps = 25
        elif difficulty == "easy":
            dim = (5, 5)
            num_boxes = 1
            search_depth = 15
            max_steps = 50
        elif difficulty == "medium":
            dim = (6, 6)
            num_boxes = 2
            search_depth = 20
            max_steps = 100
        else:  # hard
            dim = (7, 7)
            num_boxes = 3
            search_depth = 25
            max_steps = 150
        
        # Generate room using the working function
        room_fixed, room_state, _, _ = generate_room(
            dim=dim,
            initial_seed=seed,
            num_boxes=num_boxes,
            search_depth=search_depth
        )
        
        return {
            "dim_room": list(dim),
            "room_fixed": room_fixed.tolist(),
            "room_state": room_state.tolist(),
            "boxes_on_target": 0,
            "max_steps": max_steps,
            "num_boxes": num_boxes
        }
    
    def extract_achievements(self, observation: Dict[str, Any], step_num: int) -> List[str]:
        """Extract Sokoban-specific achievements"""
        achievements = []
        
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        steps_taken = observation.get("steps_taken", 0)
        
        # Movement achievements
        if steps_taken > 0:
            achievements.append("started_moving")
        if steps_taken >= 10:
            achievements.append("persistent_effort")
        if steps_taken >= 25:
            achievements.append("determined_player")
        
        # Progress achievements
        if boxes_on_target > 0:
            achievements.append(f"placed_{boxes_on_target}_boxes")
        
        if total_boxes > 1 and boxes_on_target >= total_boxes // 2:
            achievements.append("halfway_complete")
        
        # Success achievements
        if boxes_on_target == total_boxes and total_boxes > 0:
            achievements.append("puzzle_solved")
            
            # Efficiency achievements based on steps
            if steps_taken <= 15:
                achievements.append("very_efficient")
            elif steps_taken <= 25:
                achievements.append("efficient_solution")
            elif steps_taken <= 40:
                achievements.append("decent_solution")
        
        # Special achievements
        if boxes_on_target > 0 and steps_taken <= 5:
            achievements.append("quick_start")
        
        return achievements
    
    async def run_episode(self, agent, scenario_config: Dict[str, Any]) -> SokobanResult:
        """Run a single Sokoban episode with detailed tracking"""
        episode_id = f"sokoban_{scenario_config.get('name', 'default')}_{int(time.time())}"
        start_time = time.time()
        
        step_rewards = []
        step_observations = []
        achievement_progression = []
        previous_achievements = set()
        
        try:
            # Create proper room snapshot
            difficulty = scenario_config.get("difficulty", "easy")
            seed = scenario_config.get("seed", 42)
            room_snapshot = self.create_room_snapshot(difficulty, seed)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Initialize with proper room
                init_response = await client.post(
                    f"{self.service_url}/env/Sokoban/initialize",
                    json={"initial_state": room_snapshot, "config": {}}
                )
                
                if init_response.status_code != 200:
                    raise Exception(f"Failed to initialize: {init_response.text}")
                
                init_data = init_response.json()
                env_id = init_data["env_id"]
                observation = init_data["observation"]
                step_observations.append(observation.copy())
                
                # Reset agent
                await agent.reset()
                
                # Track initial state
                total_boxes = observation.get("num_boxes", 0)
                max_steps = observation.get("max_steps", 50)
                room_size = observation.get("dim_room", [0, 0])
                
                # Initial achievements
                initial_achievements = self.extract_achievements(observation, 0)
                if initial_achievements:
                    achievement_progression.append({
                        "step": 0,
                        "new_achievements": initial_achievements,
                        "total_achievements": len(initial_achievements)
                    })
                    previous_achievements.update(initial_achievements)
                
                # Episode loop
                step_count = 0
                while step_count < max_steps and not observation.get("terminated", False):
                    # Agent takes action
                    action = await agent.act(observation, step_count)
                    
                    # Send action to environment
                    step_response = await client.post(
                        f"{self.service_url}/env/Sokoban/step",
                        json={
                            "env_id": env_id,
                            "request_id": f"step_{step_count}",
                            "action": {
                                "tool_calls": [{"tool": "interact", "args": {"action": action}}]
                            }
                        }
                    )
                    
                    if step_response.status_code != 200:
                        raise Exception(f"Step {step_count} failed: {step_response.text}")
                    
                    step_data = step_response.json()
                    observation = step_data["observation"]
                    reward = step_data.get("reward", 0.0)
                    
                    step_rewards.append(reward)
                    step_observations.append(observation.copy())
                    step_count += 1
                    
                    # Track achievement progression
                    current_achievements = set(self.extract_achievements(observation, step_count))
                    new_achievements = current_achievements - previous_achievements
                    
                    if new_achievements:
                        achievement_progression.append({
                            "step": step_count,
                            "new_achievements": list(new_achievements),
                            "total_achievements": len(current_achievements)
                        })
                        previous_achievements = current_achievements
                    
                    # Check for early termination
                    if observation.get("terminated", False):
                        break
                
                # Terminate environment
                await client.post(
                    f"{self.service_url}/env/Sokoban/terminate",
                    json={"env_id": env_id}
                )
                
                # Calculate final metrics
                episode_length = time.time() - start_time
                boxes_placed = observation.get("boxes_on_target", 0)
                steps_taken = observation.get("steps_taken", step_count)
                
                success = boxes_placed == total_boxes and total_boxes > 0
                completion_rate = boxes_placed / max(total_boxes, 1)
                step_efficiency = boxes_placed / max(steps_taken, 1) if steps_taken > 0 else 0
                movement_efficiency = boxes_placed / max(step_count, 1) if step_count > 0 else 0
                final_reward = sum(step_rewards)
                all_achievements = self.extract_achievements(observation, steps_taken)
                
                return SokobanResult(
                    episode_id=episode_id,
                    scenario=scenario_config.get("name", "default"),
                    agent_name=agent.name,
                    model_name=agent.model_name,
                    success=success,
                    boxes_placed=boxes_placed,
                    total_boxes=total_boxes,
                    completion_rate=completion_rate,
                    steps_taken=steps_taken,
                    max_steps=max_steps,
                    step_efficiency=step_efficiency,
                    movement_efficiency=movement_efficiency,
                    final_reward=final_reward,
                    episode_length_seconds=episode_length,
                    room_size=room_size,
                    achievements=all_achievements,
                    achievement_progression=achievement_progression,
                    step_rewards=step_rewards,
                    step_observations=step_observations
                )
                
        except Exception as e:
            return SokobanResult(
                episode_id=episode_id,
                scenario=scenario_config.get("name", "default"),
                agent_name=getattr(agent, 'name', 'Unknown'),
                model_name=getattr(agent, 'model_name', 'Unknown'),
                success=False,
                boxes_placed=0,
                total_boxes=0,
                completion_rate=0.0,
                steps_taken=0,
                max_steps=0,
                step_efficiency=0.0,
                movement_efficiency=0.0,
                final_reward=0.0,
                episode_length_seconds=time.time() - start_time,
                room_size=[0, 0],
                achievements=[],
                achievement_progression=[],
                step_rewards=[],
                step_observations=[],
                error=str(e),
                terminated_early=True
            )

class SokobanAgent:
    """Demo agent with Sokoban-specific strategies"""
    
    def __init__(self, model_name: str = "demo-sokoban-agent", strategy: str = "systematic"):
        self.model_name = model_name
        self.name = f"SokobanAgent-{strategy}"
        self.strategy = strategy
        self.step_count = 0
        self.last_position = None
        self.stuck_count = 0
    
    async def reset(self):
        self.step_count = 0
        self.last_position = None
        self.stuck_count = 0
    
    async def act(self, observation: Dict[str, Any], step_num: int) -> int:
        """Sokoban-specific action selection"""
        self.step_count += 1
        
        current_position = observation.get("player_position", [0, 0])
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        
        # Check if stuck (same position for multiple steps)
        if self.last_position == current_position:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        
        self.last_position = current_position
        
        if self.strategy == "systematic":
            return self._systematic_strategy(observation, step_num)
        elif self.strategy == "adaptive":
            return self._adaptive_strategy(observation, step_num)
        else:
            return self._random_strategy(observation, step_num)
    
    def _systematic_strategy(self, observation: Dict[str, Any], step_num: int) -> int:
        """Systematic exploration pattern"""
        # Actions: 0=noop, 2=up, 4=left, 6=down, 8=right
        actions = [8, 8, 2, 2, 4, 4, 6, 6]  # Favor horizontal movement
        
        # If stuck, try different pattern
        if self.stuck_count > 2:
            actions = [6, 4, 2, 8]  # Try all directions
        
        return actions[step_num % len(actions)]
    
    def _adaptive_strategy(self, observation: Dict[str, Any], step_num: int) -> int:
        """Adaptive strategy based on game state"""
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        
        # If no boxes placed yet, explore more
        if boxes_on_target == 0:
            actions = [8, 2, 4, 6]  # Try all directions
        else:
            # Some progress made, be more systematic
            actions = [8, 8, 2, 6, 4]  # Favor right and up
        
        # If stuck, try opposite direction
        if self.stuck_count > 3:
            actions = [4, 6, 2, 8]  # Reverse order
        
        return actions[step_num % len(actions)]
    
    def _random_strategy(self, observation: Dict[str, Any], step_num: int) -> int:
        """Simple cyclic pattern"""
        actions = [8, 2, 4, 6]  # Right, Up, Left, Down
        return actions[step_num % len(actions)]

class SokobanBenchmark:
    """Sokoban benchmark runner and analyzer"""
    
    def __init__(self):
        self.evaluator = SokobanEvaluator()
    
    async def run_benchmark(self, scenarios: List[Dict[str, Any]], agent) -> List[SokobanResult]:
        """Run complete benchmark"""
        all_results = []
        
        total_episodes = sum(s.get("num_episodes", 3) for s in scenarios)
        episode_count = 0
        
        print(f"🎯 Sokoban Benchmark: {len(scenarios)} scenarios, {total_episodes} episodes")
        print(f"🤖 Agent: {agent.name} ({agent.model_name})")
        print("=" * 60)
        
        for scenario in scenarios:
            scenario_name = scenario.get("name", "default")
            num_episodes = scenario.get("num_episodes", 3)
            difficulty = scenario.get("difficulty", "easy")
            
            print(f"\n📋 Scenario: {scenario_name} ({difficulty}) - {num_episodes} episodes")
            
            scenario_results = []
            for episode in range(num_episodes):
                episode_count += 1
                print(f"  Episode {episode + 1}/{num_episodes} ({episode_count}/{total_episodes})...", end=" ")
                
                # Use different seeds for variety
                scenario_copy = scenario.copy()
                scenario_copy["seed"] = 42 + episode
                
                result = await self.evaluator.run_episode(agent, scenario_copy)
                scenario_results.append(result)
                all_results.append(result)
                
                if result.error:
                    print(f"❌ Error: {result.error[:40]}...")
                else:
                    status = "✅ SUCCESS" if result.success else "❌ FAILED"
                    print(f"{status} ({result.boxes_placed}/{result.total_boxes} boxes, {result.steps_taken} steps)")
                    
                    if result.achievements:
                        key_achievements = [a for a in result.achievements if a in ["puzzle_solved", "efficient_solution", "very_efficient"]]
                        if key_achievements:
                            print(f"    🏆 {', '.join(key_achievements)}")
            
            # Scenario summary
            valid_results = [r for r in scenario_results if not r.error]
            if valid_results:
                success_rate = sum(r.success for r in valid_results) / len(valid_results)
                avg_steps = sum(r.steps_taken for r in valid_results) / len(valid_results)
                avg_completion = sum(r.completion_rate for r in valid_results) / len(valid_results)
                
                print(f"  📊 Scenario Summary: {success_rate:.1%} success, {avg_steps:.1f} avg steps, {avg_completion:.1%} avg completion")
        
        return all_results
    
    def analyze_results(self, results: List[SokobanResult]):
        """Comprehensive analysis of Sokoban results"""
        valid_results = [r for r in results if not r.error]
        
        if not valid_results:
            print("\n❌ No valid results to analyze")
            return
        
        print("\n" + "="*70)
        print("🎯 SOKOBAN BENCHMARK ANALYSIS")
        print("="*70)
        
        # Overall performance
        success_rate = sum(r.success for r in valid_results) / len(valid_results)
        avg_completion = sum(r.completion_rate for r in valid_results) / len(valid_results)
        avg_steps = sum(r.steps_taken for r in valid_results) / len(valid_results)
        avg_efficiency = sum(r.step_efficiency for r in valid_results) / len(valid_results)
        
        print(f"📊 Overall Performance:")
        print(f"   • Success Rate: {success_rate:.1%} ({sum(r.success for r in valid_results)}/{len(valid_results)} episodes)")
        print(f"   • Average Completion: {avg_completion:.1%}")
        print(f"   • Average Steps: {avg_steps:.1f}")
        print(f"   • Step Efficiency: {avg_efficiency:.3f} boxes/step")
        
        # Difficulty analysis
        difficulty_stats = {}
        for result in valid_results:
            scenario = result.scenario
            if scenario not in difficulty_stats:
                difficulty_stats[scenario] = []
            difficulty_stats[scenario].append(result)
        
        print(f"\n📈 Performance by Difficulty:")
        for scenario, scenario_results in difficulty_stats.items():
            success_rate = sum(r.success for r in scenario_results) / len(scenario_results)
            avg_steps = sum(r.steps_taken for r in scenario_results) / len(scenario_results)
            avg_boxes = sum(r.total_boxes for r in scenario_results) / len(scenario_results)
            
            print(f"   • {scenario}: {success_rate:.1%} success, {avg_steps:.1f} steps, {avg_boxes:.1f} boxes")
        
        # Achievement analysis
        all_achievements = []
        for r in valid_results:
            all_achievements.extend(r.achievements)
        
        if all_achievements:
            achievement_counts = {}
            for achievement in all_achievements:
                achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
            
            print(f"\n🏅 Achievement Analysis:")
            sorted_achievements = sorted(achievement_counts.items(), key=lambda x: x[1], reverse=True)
            for achievement, count in sorted_achievements[:8]:
                percentage = count / len(valid_results) * 100
                print(f"   • {achievement}: {count} times ({percentage:.0f}% of episodes)")
        
        # Efficiency insights
        successful_results = [r for r in valid_results if r.success]
        if successful_results:
            efficient_solutions = [r for r in successful_results if "efficient_solution" in r.achievements or "very_efficient" in r.achievements]
            
            print(f"\n⚡ Efficiency Insights:")
            print(f"   • Successful Episodes: {len(successful_results)}")
            print(f"   • Efficient Solutions: {len(efficient_solutions)} ({len(efficient_solutions)/len(successful_results):.1%})")
            
            if efficient_solutions:
                avg_efficient_steps = sum(r.steps_taken for r in efficient_solutions) / len(efficient_solutions)
                print(f"   • Average Steps (Efficient): {avg_efficient_steps:.1f}")
    
    def save_detailed_report(self, results: List[SokobanResult], filepath: str):
        """Save comprehensive report"""
        report_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_episodes": len(results),
                "valid_episodes": len([r for r in results if not r.error]),
                "agent_name": results[0].agent_name if results else "Unknown",
                "model_name": results[0].model_name if results else "Unknown"
            },
            "summary": {
                "success_rate": sum(r.success for r in results if not r.error) / max(len([r for r in results if not r.error]), 1),
                "average_completion_rate": sum(r.completion_rate for r in results if not r.error) / max(len([r for r in results if not r.error]), 1),
                "average_steps": sum(r.steps_taken for r in results if not r.error) / max(len([r for r in results if not r.error]), 1)
            },
            "detailed_results": [asdict(r) for r in results]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to {filepath}")

# Benchmark scenarios
SOKOBAN_SCENARIOS = [
    {
        "name": "tutorial",
        "difficulty": "tutorial",
        "num_episodes": 5,
        "description": "Simple 4x4 puzzles with 1 box"
    },
    {
        "name": "easy",
        "difficulty": "easy",
        "num_episodes": 4,
        "description": "Basic 5x5 puzzles with 1 box"
    },
    {
        "name": "medium",
        "difficulty": "medium",
        "num_episodes": 3,
        "description": "Intermediate 6x6 puzzles with 2 boxes"
    },
    {
        "name": "hard",
        "difficulty": "hard",
        "num_episodes": 2,
        "description": "Advanced 7x7 puzzles with 3 boxes"
    }
]

async def main():
    """Run focused Sokoban evaluation"""
    print("🎯 Focused Sokoban Evaluation")
    print("Testing spatial reasoning and planning capabilities")
    print("=" * 60)
    
    # Check service health
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get("http://localhost:8001/health")
            if health.status_code == 200:
                supported = health.json()["supported_environments"]
                if "Sokoban" in supported:
                    print("✅ Sokoban service is available")
                else:
                    print("❌ Sokoban not supported")
                    return
            else:
                print("❌ Service not responding")
                return
    except Exception as e:
        print(f"❌ Service connection failed: {e}")
        return
    
    # Test different agent strategies
    agents = [
        SokobanAgent("systematic-v1", "systematic"),
        SokobanAgent("adaptive-v1", "adaptive")
    ]
    
    benchmark = SokobanBenchmark()
    
    for agent in agents:
        print(f"\n🤖 Testing Agent: {agent.name}")
        print("-" * 40)
        
        # Run benchmark
        results = await benchmark.run_benchmark(SOKOBAN_SCENARIOS, agent)
        
        # Analyze results
        benchmark.analyze_results(results)
        
        # Save report
        agent_name_clean = agent.name.replace("-", "_").lower()
        report_path = f"results/sokoban_{agent_name_clean}_report.json"
        benchmark.save_detailed_report(results, report_path)
    
    print("\n🎉 Sokoban evaluation complete!")
    print("\n💡 Key insights:")
    print("   • Spatial reasoning evaluation through box-pushing puzzles")
    print("   • Planning depth measurement via step efficiency")
    print("   • Difficulty scaling from 4x4 to 7x7 rooms")
    print("   • Achievement-based progress tracking")
    print("   • Comprehensive strategy comparison")

if __name__ == "__main__":
    asyncio.run(main()) 