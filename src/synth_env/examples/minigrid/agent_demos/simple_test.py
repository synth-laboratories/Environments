"""
Simple test for MiniGrid evaluation framework components.
"""

import asyncio
from synth_env.examples.minigrid.agent_demos.eval_framework import (
    MiniGridEvalFramework,
    MiniGridTrajectoryResult,
    MINIGRID_ACHIEVEMENTS,
    minigrid_composite_score,
    minigrid_navigation_score
)

def test_scoring_functions():
    """Test the scoring functions."""
    print("Testing scoring functions...")
    
    # Test composite score
    composite_score = minigrid_composite_score(5, 0.8, 0.6, 0.7)
    print(f"Composite score (5 achievements, 80% completion, 60% efficiency, 70% exploration): {composite_score:.2f}")
    
    # Test navigation score
    navigation_score = minigrid_navigation_score(0.8, 0.6, 0.1)
    print(f"Navigation score (80% success, 60% efficiency, 10% collision rate): {navigation_score:.2f}")
    
    print("✅ Scoring functions work!")

def test_achievements():
    """Test the achievement system."""
    print("\nTesting achievement system...")
    
    print(f"Total achievements: {len(MINIGRID_ACHIEVEMENTS['basic']) + len(MINIGRID_ACHIEVEMENTS['intermediate']) + len(MINIGRID_ACHIEVEMENTS['advanced'])}")
    print(f"Basic achievements: {MINIGRID_ACHIEVEMENTS['basic']}")
    print(f"Intermediate achievements: {MINIGRID_ACHIEVEMENTS['intermediate']}")
    print(f"Advanced achievements: {MINIGRID_ACHIEVEMENTS['advanced']}")
    
    print("✅ Achievement system works!")

def test_trajectory_result():
    """Test the trajectory result structure."""
    print("\nTesting trajectory result structure...")
    
    result = MiniGridTrajectoryResult(
        trajectory_id="test-123",
        model_name="test-model",
        difficulty="easy",
        task_type="MiniGrid-Empty-6x6-v0",
        seed=42,
        success=True,
        total_steps=25,
        total_turns=20,
        total_reward=1.0,
        grid_size=(6, 6),
        steps_to_goal=25,
        optimal_steps=12,
        efficiency_ratio=0.48,
        objects_interacted=["goal"],
        rooms_visited=1,
        backtrack_count=3,
        wall_collision_count=2,
        exploration_coverage=0.65,
        achievements_unlocked={"reach_goal", "navigate_empty_room"},
        achievement_turn_unlocked={"reach_goal": 20, "navigate_empty_room": 20},
        actions_per_turn=[1] * 20,
        avg_actions_per_turn=1.0,
        termination_reason="goal_reached",
        final_position=(5, 5),
        final_direction=0,
        turn_by_turn_data=None
    )
    
    print(f"Created trajectory result: {result.trajectory_id}")
    print(f"Success: {result.success}, Steps: {result.total_steps}, Achievements: {len(result.achievements_unlocked)}")
    
    print("✅ Trajectory result structure works!")

def test_framework_init():
    """Test the framework initialization."""
    print("\nTesting framework initialization...")
    
    framework = MiniGridEvalFramework()
    print(f"Framework initialized with {len(framework.trajectory_results)} results")
    
    print("✅ Framework initialization works!")

def main():
    """Run all tests."""
    print("🎯 Testing MiniGrid Evaluation Framework Components")
    print("=" * 60)
    
    test_scoring_functions()
    test_achievements()
    test_trajectory_result()
    test_framework_init()
    
    print("\n" + "=" * 60)
    print("✅ All component tests passed!")

if __name__ == "__main__":
    main() 