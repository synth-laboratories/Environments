#!/usr/bin/env python3
"""
Concrete implementation example of the standardized evaluation framework.
This shows how to run Gemini-1.5-Flash ReAct agents across multiple environments.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path
import toml

# ============================================================================
# Core Framework Classes
# ============================================================================

@dataclass
class EvalResult:
    """Standardized result format for any environment evaluation"""
    env_name: str
    agent_name: str
    model_name: str
    episode_id: str
    
    # Core metrics (standardized across all envs)
    success: bool
    final_reward: float
    total_steps: int
    episode_length_seconds: float
    
    # Environment-specific metrics
    env_specific_metrics: Dict[str, Any]
    
    # Intermediate tracking
    step_rewards: List[float]
    achievements: List[str]
    intermediate_states: List[Dict[str, Any]]
    
    # Error handling
    error: Optional[str] = None
    terminated_early: bool = False

class BaseEnvironmentEvaluator(ABC):
    """Base class for environment-specific evaluators"""
    
    def __init__(self, env_name: str, service_url: str = "http://localhost:8001"):
        self.env_name = env_name
        self.service_url = service_url
    
    @abstractmethod
    async def create_task_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create environment-specific task instance"""
        pass
    
    @abstractmethod
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        """Extract environment-specific achievements from observation"""
        pass
    
    @abstractmethod
    def calculate_success(self, final_obs: Dict[str, Any]) -> bool:
        """Determine if episode was successful"""
        pass
    
    @abstractmethod
    def get_env_specific_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environment-specific metrics"""
        pass
    
    async def run_episode(self, agent, config: Dict[str, Any]) -> EvalResult:
        """Run a single episode with the given agent"""
        episode_id = f"{self.env_name}_{int(time.time())}"
        start_time = time.time()
        
        step_rewards = []
        achievements_history = []
        intermediate_states = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Create task instance
                task_instance = await self.create_task_instance(config)
                
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
                
                # 3. Reset agent for new episode
                await agent.reset()
                
                # 4. Run episode loop
                total_steps = 0
                max_steps = config.get("max_steps", 100)
                
                while total_steps < max_steps:
                    # Extract current achievements
                    current_achievements = self.extract_achievements(observation)
                    achievements_history.extend([a for a in current_achievements if a not in achievements_history])
                    
                    # Store intermediate state
                    intermediate_states.append({
                        "step": total_steps,
                        "observation": observation.copy(),
                        "achievements": current_achievements.copy()
                    })
                    
                    # Check termination
                    if observation.get("terminated", False):
                        break
                    
                    # Get available tools from observation
                    available_tools = observation.get("tools", [])
                    
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
                
                # 5. Terminate environment
                await client.post(
                    f"{self.service_url}/env/{self.env_name}/terminate",
                    json={"env_id": env_id}
                )
                
                # 6. Calculate final metrics
                episode_length = time.time() - start_time
                success = self.calculate_success(observation)
                final_reward = sum(step_rewards)
                
                episode_data = {
                    "final_observation": observation,
                    "step_rewards": step_rewards,
                    "total_steps": total_steps
                }
                env_specific_metrics = self.get_env_specific_metrics(episode_data)
                
                return EvalResult(
                    env_name=self.env_name,
                    agent_name=agent.name,
                    model_name=agent.model_name,
                    episode_id=episode_id,
                    success=success,
                    final_reward=final_reward,
                    total_steps=total_steps,
                    episode_length_seconds=episode_length,
                    env_specific_metrics=env_specific_metrics,
                    step_rewards=step_rewards,
                    achievements=achievements_history,
                    intermediate_states=intermediate_states
                )
                
        except Exception as e:
            return EvalResult(
                env_name=self.env_name,
                agent_name=agent.name,
                model_name=agent.model_name,
                episode_id=episode_id,
                success=False,
                final_reward=0.0,
                total_steps=0,
                episode_length_seconds=time.time() - start_time,
                env_specific_metrics={},
                step_rewards=[],
                achievements=[],
                intermediate_states=[],
                error=str(e),
                terminated_early=True
            )

# ============================================================================
# Environment-Specific Evaluators
# ============================================================================

class SokobanEvaluator(BaseEnvironmentEvaluator):
    def __init__(self):
        super().__init__("Sokoban")
    
    async def create_task_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Sokoban room with specified difficulty"""
        # For demo purposes, use a simple fixed room
        # In practice, you'd import and use the room generation utils
        
        if config.get("difficulty") == "easy":
            # Simple 5x5 room with 1 box
            return {
                "dim_room": [5, 5],
                "room_fixed": [[1,1,1,1,1],[1,0,0,0,1],[1,0,3,0,1],[1,0,0,0,1],[1,1,1,1,1]],
                "room_state": [[0,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,4,0,0,0],[0,0,0,0,0]],
                "boxes_on_target": 0,
                "max_steps": config.get("max_steps", 50),
                "num_boxes": 1
            }
        else:
            # Medium difficulty room
            return {
                "dim_room": [6, 6],
                "room_fixed": [[1,1,1,1,1,1],[1,0,0,0,0,1],[1,0,3,3,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,1,1,1,1,1]],
                "room_state": [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,2,2,0,0],[0,0,0,0,0,0],[0,4,0,0,0,0],[0,0,0,0,0,0]],
                "boxes_on_target": 0,
                "max_steps": config.get("max_steps", 100),
                "num_boxes": 2
            }
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        achievements = []
        boxes_on_target = observation.get("boxes_on_target", 0)
        total_boxes = observation.get("num_boxes", 0)
        steps_taken = observation.get("steps_taken", 0)
        
        if boxes_on_target > 0:
            achievements.append(f"placed_{boxes_on_target}_boxes")
        if boxes_on_target == total_boxes:
            achievements.append("puzzle_solved")
        if steps_taken > 0 and steps_taken <= 20:
            achievements.append("efficient_solution")
        if steps_taken > 0:
            achievements.append("made_progress")
        
        return achievements
    
    def calculate_success(self, final_obs: Dict[str, Any]) -> bool:
        return (final_obs.get("boxes_on_target", 0) == 
                final_obs.get("num_boxes", 0))
    
    def get_env_specific_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        final_obs = episode_data["final_observation"]
        return {
            "boxes_placed": final_obs.get("boxes_on_target", 0),
            "total_boxes": final_obs.get("num_boxes", 0),
            "placement_efficiency": final_obs.get("boxes_on_target", 0) / max(final_obs.get("num_boxes", 1), 1),
            "steps_taken": final_obs.get("steps_taken", 0),
            "room_size": final_obs.get("dim_room", [0, 0])
        }

class TicTacToeEvaluator(BaseEnvironmentEvaluator):
    def __init__(self):
        super().__init__("TicTacToe")
    
    async def create_task_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # TicTacToe doesn't need special initialization
        return {}
    
    def extract_achievements(self, observation: Dict[str, Any]) -> List[str]:
        achievements = []
        winner = observation.get("winner")
        
        if winner == "X":
            achievements.append("player_won")
        elif winner == "draw":
            achievements.append("forced_draw")
        elif winner == "O":
            achievements.append("opponent_won")
        elif winner is None and observation.get("moves_made", 0) > 0:
            achievements.append("game_in_progress")
        
        return achievements
    
    def calculate_success(self, final_obs: Dict[str, Any]) -> bool:
        return final_obs.get("winner") == "X"  # Assuming agent plays as X
    
    def get_env_specific_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        final_obs = episode_data["final_observation"]
        return {
            "winner": final_obs.get("winner"),
            "game_completed": final_obs.get("terminated", False),
            "moves_made": final_obs.get("moves_made", 0),
            "board_state": final_obs.get("board", [])
        }

# ============================================================================
# Simple ReAct Agent Implementation
# ============================================================================

class SimpleReActAgent:
    """Simple ReAct agent for demonstration"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.name = "SimpleReActAgent"
        self.conversation_history = []
    
    async def reset(self):
        """Reset agent state for new episode"""
        self.conversation_history = []
    
    async def act(self, observation: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
        """Take an action given an observation"""
        # For demo purposes, implement simple heuristics
        # In practice, this would use an LLM with ReAct prompting
        
        env_name = observation.get("env_name", "Unknown")
        
        if "Sokoban" in str(observation):
            return self._sokoban_action(observation)
        elif "TicTacToe" in str(observation) or "board" in observation:
            return self._tictactoe_action(observation)
        else:
            # Default action
            if "interact" in available_tools:
                return {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
            else:
                return {"tool_calls": [{"tool": available_tools[0], "args": {}}]}
    
    def _sokoban_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Sokoban strategy: try to move right, then down, then left, then up"""
        steps_taken = observation.get("steps_taken", 0)
        actions = [8, 2, 4, 6]  # right, down, left, up
        action = actions[steps_taken % len(actions)]
        return {"tool_calls": [{"tool": "interact", "args": {"action": action}}]}
    
    def _tictactoe_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Simple TicTacToe strategy: play in first available spot"""
        board = observation.get("board", [])
        
        # Find first empty spot
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == " ":
                    position = f"{chr(ord('A') + i)}{j + 1}"
                    return {"tool_calls": [{"tool": "interact", "args": {"action": position}}]}
        
        # Fallback
        return {"tool_calls": [{"tool": "interact", "args": {"action": "A1"}}]}

# ============================================================================
# Evaluation Runner
# ============================================================================

@dataclass
class BenchmarkResults:
    """Aggregated results across environments"""
    environment_results: Dict[str, List[EvalResult]]
    
    def __init__(self):
        self.environment_results = {}
    
    def add_environment_results(self, env_name: str, results: List[EvalResult]):
        self.environment_results[env_name] = results
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {}
        
        for env_name, results in self.environment_results.items():
            if not results:
                continue
                
            success_rate = sum(r.success for r in results) / len(results)
            avg_reward = sum(r.final_reward for r in results) / len(results)
            avg_steps = sum(r.total_steps for r in results) / len(results)
            
            # Aggregate achievements
            all_achievements = []
            for r in results:
                all_achievements.extend(r.achievements)
            achievement_counts = {}
            for achievement in all_achievements:
                achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
            
            # Aggregate environment-specific metrics
            env_metrics = {}
            if results:
                for key in results[0].env_specific_metrics.keys():
                    values = [r.env_specific_metrics.get(key, 0) for r in results]
                    if all(isinstance(v, (int, float)) for v in values):
                        env_metrics[f"avg_{key}"] = sum(values) / len(values)
            
            summary[env_name] = {
                "success_rate": success_rate,
                "average_reward": avg_reward,
                "average_steps": avg_steps,
                "total_episodes": len(results),
                "achievement_breakdown": achievement_counts,
                "env_specific_metrics": env_metrics
            }
        
        return summary
    
    def print_summary(self):
        """Print formatted summary to console"""
        summary = self.get_summary_stats()
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        for env_name, stats in summary.items():
            print(f"\n{env_name.upper()} ENVIRONMENT:")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Average Reward: {stats['average_reward']:.2f}")
            print(f"  Average Steps: {stats['average_steps']:.1f}")
            print(f"  Total Episodes: {stats['total_episodes']}")
            
            if stats['achievement_breakdown']:
                print(f"\n  Achievement Breakdown:")
                for achievement, count in stats['achievement_breakdown'].items():
                    percentage = count / stats['total_episodes'] * 100
                    print(f"    {achievement}: {count}/{stats['total_episodes']} ({percentage:.1f}%)")
            
            if stats['env_specific_metrics']:
                print(f"\n  Environment-Specific Metrics:")
                for metric, value in stats['env_specific_metrics'].items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.2f}")
                    else:
                        print(f"    {metric}: {value}")
    
    def save_to_json(self, filepath: str):
        """Save detailed results to JSON file"""
        data = {
            "summary": self.get_summary_stats(),
            "detailed_results": {}
        }
        
        for env_name, results in self.environment_results.items():
            data["detailed_results"][env_name] = [asdict(r) for r in results]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")

class EvaluationRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluators = self._setup_evaluators()
        self.agent = self._setup_agent()
    
    def _setup_evaluators(self) -> Dict[str, BaseEnvironmentEvaluator]:
        """Initialize environment evaluators"""
        evaluators = {
            "Sokoban": SokobanEvaluator(),
            "TicTacToe": TicTacToeEvaluator(),
        }
        return evaluators
    
    def _setup_agent(self):
        """Initialize agent from config"""
        agent_config = self.config.get("agent", {})
        model_name = agent_config.get("model_name", "gemini-1.5-flash")
        return SimpleReActAgent(model_name=model_name)
    
    async def run_benchmark(self) -> BenchmarkResults:
        """Run complete benchmark across all environments/scenarios"""
        results = BenchmarkResults()
        
        environments = self.config.get("evaluation", {}).get("environments", ["Sokoban"])
        
        for env_name in environments:
            if env_name in self.evaluators:
                print(f"\n🚀 Running {env_name} evaluation...")
                env_results = await self._run_environment_benchmark(env_name)
                results.add_environment_results(env_name, env_results)
            else:
                print(f"⚠️ Evaluator for {env_name} not implemented yet")
        
        return results
    
    async def _run_environment_benchmark(self, env_name: str) -> List[EvalResult]:
        """Run benchmark for a specific environment"""
        evaluator = self.evaluators[env_name]
        scenarios = self.config.get("scenarios", [{"name": "default"}])
        
        all_results = []
        for scenario in scenarios:
            print(f"  Running scenario: {scenario.get('name', 'default')}")
            scenario_results = await self._run_scenario(evaluator, scenario)
            all_results.extend(scenario_results)
        
        return all_results
    
    async def _run_scenario(self, evaluator: BaseEnvironmentEvaluator, scenario: Dict[str, Any]) -> List[EvalResult]:
        """Run multiple episodes for a scenario"""
        num_episodes = scenario.get("num_episodes", 3)
        
        results = []
        for episode in range(num_episodes):
            print(f"    Episode {episode + 1}/{num_episodes}...")
            result = await evaluator.run_episode(self.agent, scenario)
            results.append(result)
            
            # Print episode result
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"      {status} - Reward: {result.final_reward:.2f}, Steps: {result.total_steps}")
            if result.error:
                print(f"      Error: {result.error}")
        
        return results

# ============================================================================
# Demo Configuration and Main Function
# ============================================================================

DEMO_CONFIG = {
    "metadata": {
        "name": "Demo Multi-Environment Benchmark",
        "description": "Test evaluation across Sokoban and TicTacToe"
    },
    "agent": {
        "type": "SimpleReActAgent",
        "model_name": "gemini-1.5-flash"
    },
    "evaluation": {
        "environments": ["Sokoban", "TicTacToe"],
        "parallel_environments": 1
    },
    "scenarios": [
        {
            "name": "easy_sokoban",
            "difficulty": "easy",
            "max_steps": 30,
            "num_episodes": 3
        },
        {
            "name": "medium_sokoban", 
            "difficulty": "medium",
            "max_steps": 50,
            "num_episodes": 2
        }
    ]
}

async def main():
    """Run the demonstration evaluation"""
    print("🎯 Starting Standardized Environment Evaluation Demo")
    print("=" * 60)
    
    # Check if service is running
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get("http://localhost:8001/health")
            if health.status_code == 200:
                print("✅ Environment service is running")
                supported = health.json()["supported_environments"]
                print(f"   Supported environments: {supported}")
            else:
                print("❌ Environment service not responding")
                return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("   Make sure to start the service with:")
        print("   uv run uvicorn src.synth_env.service.app:app --reload --host 0.0.0.0 --port 8001")
        return
    
    # Run evaluation
    runner = EvaluationRunner(DEMO_CONFIG)
    results = await runner.run_benchmark()
    
    # Print summary
    results.print_summary()
    
    # Save results
    results.save_to_json("results/demo_evaluation_results.json")
    
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main()) 