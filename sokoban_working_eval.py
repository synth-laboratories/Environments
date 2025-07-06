#!/usr/bin/env python3
"""
Working Sokoban Evaluation - Using Proven Parameters
This uses the exact room generation parameters that work in the test suite.
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
    success: bool
    boxes_placed: int
    total_boxes: int
    completion_rate: float
    steps_taken: int
    max_steps: int
    step_efficiency: float
    final_reward: float
    episode_length_seconds: float
    room_size: List[int]
    achievements: List[str]
    achievement_progression: List[Dict[str, Any]]
    step_rewards: List[float]
    error: Optional[str] = None

class SokobanEvaluator:
    """Working Sokoban evaluator with proven room generation"""
    
    def __init__(self, service_url: str = "http://localhost:8001"):
        self.service_url = service_url
    
    def create_room_snapshot(self, difficulty: str, seed: int = 42) -> Dict[str, Any]:
        """Create valid Sokoban room using PROVEN working parameters"""
        from synth_env.examples.sokoban.engine_helpers.room_utils import generate_room
        
        # Use ONLY the parameters that work reliably
        if difficulty == "easy":
            # Exactly from working test
            dim = (5, 5)
            num_boxes = 1
            search_depth = 15
            max_steps = 50
        elif difficulty == "medium":
            # Proven to work from test
            dim = (6, 6)
            num_boxes = 2
            search_depth = 20
            max_steps = 100
        else:  # hard
            # Scaling up from working parameters
            dim = (7, 7)
            num_boxes = 2  # Keep conservative
            search_depth = 25
            max_steps = 150
        
        # Generate room using the EXACT working approach
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
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        """Extract Sokoban achievements"""
        achievements = []
        
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        steps_taken = observation.get("steps_taken", 0)
        
        # Movement achievements
        if steps_taken > 0:
            achievements.append("started_moving")
        if steps_taken >= 10:
            achievements.append("persistent_effort")
        
        # Progress achievements
        if boxes_on_target > 0:
            achievements.append(f"placed_{boxes_on_target}_boxes")
        
        # Success achievements
        if boxes_on_target == total_boxes and total_boxes > 0:
            achievements.append("puzzle_solved")
            
            # Efficiency achievements
            if steps_taken <= 20:
                achievements.append("very_efficient")
            elif steps_taken <= 30:
                achievements.append("efficient_solution")
        
        return achievements
    
    async def run_episode(self, agent, scenario_config: Dict[str, Any]) -> SokobanResult:
        """Run a single Sokoban episode"""
        episode_id = f"sokoban_{scenario_config.get('name', 'default')}_{int(time.time())}"
        start_time = time.time()
        
        step_rewards = []
        achievement_progression = []
        previous_achievements = set()
        
        try:
            # Create room snapshot
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
                
                # Reset agent
                await agent.reset()
                
                # Get episode parameters
                total_boxes = observation.get("num_boxes", 0)
                max_steps = observation.get("max_steps", 50)
                room_size = observation.get("dim_room", [0, 0])
                
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
                    step_count += 1
                    
                    # Track achievements
                    current_achievements = set(self.extract_achievements(observation))
                    new_achievements = current_achievements - previous_achievements
                    
                    if new_achievements:
                        achievement_progression.append({
                            "step": step_count,
                            "new_achievements": list(new_achievements),
                            "total_achievements": len(current_achievements)
                        })
                        previous_achievements = current_achievements
                    
                    if observation.get("terminated", False):
                        break
                
                # Terminate environment
                await client.post(
                    f"{self.service_url}/env/Sokoban/terminate",
                    json={"env_id": env_id}
                )
                
                # Calculate metrics
                episode_length = time.time() - start_time
                boxes_placed = observation.get("boxes_on_target", 0)
                steps_taken = observation.get("steps_taken", step_count)
                
                success = boxes_placed == total_boxes and total_boxes > 0
                completion_rate = boxes_placed / max(total_boxes, 1)
                step_efficiency = boxes_placed / max(steps_taken, 1) if steps_taken > 0 else 0
                final_reward = sum(step_rewards)
                all_achievements = self.extract_achievements(observation)
                
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
                    final_reward=final_reward,
                    episode_length_seconds=episode_length,
                    room_size=room_size,
                    achievements=all_achievements,
                    achievement_progression=achievement_progression,
                    step_rewards=step_rewards
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
                final_reward=0.0,
                episode_length_seconds=time.time() - start_time,
                room_size=[0, 0],
                achievements=[],
                achievement_progression=[],
                step_rewards=[],
                error=str(e)
            )

class SokobanAgent:
    """Demo agent with improved Sokoban strategy"""
    
    def __init__(self, model_name: str = "demo-agent"):
        self.model_name = model_name
        self.name = "SokobanAgent"
        self.step_count = 0
        self.last_box_count = 0
        self.exploration_phase = True
    
    async def reset(self):
        self.step_count = 0
        self.last_box_count = 0
        self.exploration_phase = True
    
    async def act(self, observation: Dict[str, Any], step_num: int) -> int:
        """Improved Sokoban strategy"""
        self.step_count += 1
        
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        
        # Check if we made progress
        if boxes_on_target > self.last_box_count:
            self.last_box_count = boxes_on_target
            self.exploration_phase = False  # Switch to exploitation
        
        # Actions: 0=noop, 2=up, 4=left, 6=down, 8=right
        if self.exploration_phase:
            # Exploration: try to find boxes and targets
            actions = [8, 8, 2, 2, 4, 4, 6, 6]  # Favor horizontal movement
        else:
            # Exploitation: more systematic approach
            actions = [8, 2, 4, 6, 8, 2, 4, 6]  # Balanced movement
        
        # Add some variety based on step count
        if self.step_count % 20 == 0:
            # Occasionally try different pattern
            actions = [4, 6, 8, 2]
        
        return actions[self.step_count % len(actions)]

# Working scenarios using PROVEN parameters
WORKING_SCENARIOS = [
    {
        "name": "easy",
        "difficulty": "easy",
        "num_episodes": 5,
        "description": "5x5 rooms with 1 box (proven to work)"
    },
    {
        "name": "medium", 
        "difficulty": "medium",
        "num_episodes": 4,
        "description": "6x6 rooms with 2 boxes (proven to work)"
    },
    {
        "name": "hard",
        "difficulty": "hard",
        "num_episodes": 3,
        "description": "7x7 rooms with 2 boxes (conservative scaling)"
    }
]

async def main():
    """Run working Sokoban evaluation"""
    print("🎯 Working Sokoban Evaluation")
    print("Using PROVEN room generation parameters")
    print("=" * 60)
    
    # Check service
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get("http://localhost:8001/health")
            if health.status_code == 200:
                supported = health.json()["supported_environments"]
                if "Sokoban" in supported:
                    print("✅ Sokoban service available")
                else:
                    print("❌ Sokoban not supported")
                    return
            else:
                print("❌ Service not responding")
                return
    except Exception as e:
        print(f"❌ Service connection failed: {e}")
        return
    
    # Run evaluation
    evaluator = SokobanEvaluator()
    agent = SokobanAgent("working-demo-v1")
    
    total_episodes = sum(s.get("num_episodes", 3) for s in WORKING_SCENARIOS)
    print(f"🤖 Agent: {agent.name} ({agent.model_name})")
    print(f"📊 Plan: {len(WORKING_SCENARIOS)} scenarios, {total_episodes} episodes")
    
    all_results = []
    episode_count = 0
    
    for scenario in WORKING_SCENARIOS:
        scenario_name = scenario.get("name", "default")
        num_episodes = scenario.get("num_episodes", 3)
        difficulty = scenario.get("difficulty", "easy")
        
        print(f"\n📋 Scenario: {scenario_name} ({difficulty})")
        print(f"   {scenario.get('description', '')}")
        
        scenario_results = []
        for episode in range(num_episodes):
            episode_count += 1
            print(f"  Episode {episode + 1}/{num_episodes} ({episode_count}/{total_episodes})...", end=" ")
            
            # Use different seeds for variety
            scenario_copy = scenario.copy()
            scenario_copy["seed"] = 42 + episode * 10
            
            result = await evaluator.run_episode(agent, scenario_copy)
            scenario_results.append(result)
            all_results.append(result)
            
            if result.error:
                print(f"❌ Error: {result.error[:50]}...")
            else:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"{status} ({result.boxes_placed}/{result.total_boxes} boxes, {result.steps_taken} steps)")
                
                if result.achievements:
                    key_achievements = [a for a in result.achievements if "puzzle_solved" in a or "efficient" in a]
                    if key_achievements:
                        print(f"    🏆 {', '.join(key_achievements)}")
        
        # Scenario summary
        valid_results = [r for r in scenario_results if not r.error]
        if valid_results:
            success_rate = sum(r.success for r in valid_results) / len(valid_results)
            avg_steps = sum(r.steps_taken for r in valid_results) / len(valid_results)
            avg_completion = sum(r.completion_rate for r in valid_results) / len(valid_results)
            
            print(f"  📊 Summary: {success_rate:.1%} success, {avg_steps:.1f} avg steps, {avg_completion:.1%} completion")
    
    # Overall analysis
    valid_results = [r for r in all_results if not r.error]
    if valid_results:
        print("\n" + "="*60)
        print("🎯 OVERALL RESULTS")
        print("="*60)
        
        success_rate = sum(r.success for r in valid_results) / len(valid_results)
        avg_completion = sum(r.completion_rate for r in valid_results) / len(valid_results)
        avg_steps = sum(r.steps_taken for r in valid_results) / len(valid_results)
        
        print(f"📊 Performance:")
        print(f"   • Success Rate: {success_rate:.1%} ({sum(r.success for r in valid_results)}/{len(valid_results)})")
        print(f"   • Average Completion: {avg_completion:.1%}")
        print(f"   • Average Steps: {avg_steps:.1f}")
        
        # Achievement analysis
        all_achievements = []
        for r in valid_results:
            all_achievements.extend(r.achievements)
        
        if all_achievements:
            achievement_counts = {}
            for achievement in all_achievements:
                achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
            
            print(f"\n🏅 Top Achievements:")
            sorted_achievements = sorted(achievement_counts.items(), key=lambda x: x[1], reverse=True)
            for achievement, count in sorted_achievements[:5]:
                percentage = count / len(valid_results) * 100
                print(f"   • {achievement}: {count} times ({percentage:.0f}%)")
        
        # Save results
        Path("results").mkdir(exist_ok=True)
        with open("results/sokoban_working_evaluation.json", "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        
        print(f"\n💾 Results saved to results/sokoban_working_evaluation.json")
    
    print("\n🎉 Working Sokoban evaluation complete!")
    print("\n💡 This demonstrates:")
    print("   • Proper room generation with proven parameters")
    print("   • Standardized evaluation metrics")
    print("   • Achievement tracking and progression")
    print("   • Comprehensive performance analysis")

if __name__ == "__main__":
    asyncio.run(main()) 