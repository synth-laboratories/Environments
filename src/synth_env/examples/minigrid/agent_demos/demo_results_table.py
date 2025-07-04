#!/usr/bin/env python3
"""
Demo showing what Gemini 1.5 Flash evaluation results would look like.
This creates mock data to demonstrate the table format.
"""

import pandas as pd
from eval_framework import (
    MiniGridAggregateResults, 
    get_pure_success_scores, 
    print_pure_success_summary
)

def create_mock_results():
    """Create mock evaluation results for demonstration."""
    
    # Mock results for Gemini 1.5 Flash on different conditions
    mock_results = [
        MiniGridAggregateResults(
            model_name="gemini-1.5-flash-latest",
            difficulty="easy",
            num_trajectories=5,
            success_rate=0.80,  # 80% success rate
            avg_total_steps=12.4,
            avg_total_turns=8.2,
            avg_total_reward=0.85,
            task_completion_rates={"Empty-5x5-v0": 0.80, "DoorKey-5x5-v0": 0.80},
            avg_efficiency_ratio=0.65,
            avg_exploration_coverage=0.72,
            avg_wall_collisions=2.1,
            avg_backtrack_count=1.8,
            unique_achievements_unlocked={"Reach Goal", "First Pickup", "Navigate Empty Room", "Door Key Master"},
            total_achievement_count=16,
            avg_achievements_per_trajectory=3.2,
            achievement_unlock_rates={
                "Reach Goal": 0.80,
                "First Pickup": 0.60,
                "Navigate Empty Room": 0.80,
                "Door Key Master": 0.40
            },
            composite_score_avg=68.5,
            composite_score_best=85.2,
            navigation_score_avg=72.3,
            navigation_score_best=89.1,
            avg_actions_per_turn_overall=1.0,
            actions_per_turn_distribution={1: 41},
            termination_breakdown={
                "goal_reached": 0.80,
                "timeout": 0.20,
                "agent_quit": 0.0,
                "environment_error": 0.0
            },
            avg_final_position=(3.2, 2.8)
        ),
        MiniGridAggregateResults(
            model_name="gemini-1.5-flash-latest",
            difficulty="medium",
            num_trajectories=5,
            success_rate=0.60,  # 60% success rate
            avg_total_steps=18.7,
            avg_total_turns=12.3,
            avg_total_reward=0.62,
            task_completion_rates={"Empty-5x5-v0": 0.60, "DoorKey-5x5-v0": 0.60},
            avg_efficiency_ratio=0.48,
            avg_exploration_coverage=0.83,
            avg_wall_collisions=3.4,
            avg_backtrack_count=2.9,
            unique_achievements_unlocked={"Reach Goal", "First Pickup", "Navigate Empty Room"},
            total_achievement_count=12,
            avg_achievements_per_trajectory=2.4,
            achievement_unlock_rates={
                "Reach Goal": 0.60,
                "First Pickup": 0.40,
                "Navigate Empty Room": 0.60,
                "Door Key Master": 0.20
            },
            composite_score_avg=52.8,
            composite_score_best=71.4,
            navigation_score_avg=58.7,
            navigation_score_best=76.2,
            avg_actions_per_turn_overall=1.0,
            actions_per_turn_distribution={1: 61},
            termination_breakdown={
                "goal_reached": 0.60,
                "timeout": 0.40,
                "agent_quit": 0.0,
                "environment_error": 0.0
            },
            avg_final_position=(2.8, 3.1)
        )
    ]
    
    return mock_results

def display_results_table():
    """Display the results in a nice table format."""
    
    print("🚀 Gemini 1.5 Flash MiniGrid Evaluation Results")
    print("="*80)
    print("📊 Mock Results (5 trajectories per condition)")
    print("Tasks: Empty-5x5-v0, DoorKey-5x5-v0")
    print("="*80)
    
    # Get mock results
    mock_results = create_mock_results()
    
    # Display pure success summary
    print_pure_success_summary(mock_results)
    
    # Create summary table
    print("\n📊 DETAILED EVALUATION SUMMARY")
    print("="*80)
    
    data = []
    for agg in mock_results:
        data.append({
            "Model": agg.model_name,
            "Difficulty": agg.difficulty,
            "✓ Success Rate": f"{agg.success_rate:.1%}",
            "Composite Score": f"{agg.composite_score_avg:.1f}",
            "Navigation Score": f"{agg.navigation_score_avg:.1f}",
            "Avg Steps": f"{agg.avg_total_steps:.1f}",
            "Avg Turns": f"{agg.avg_total_turns:.1f}",
            "Efficiency": f"{agg.avg_efficiency_ratio:.2f}",
            "Exploration": f"{agg.avg_exploration_coverage:.1%}",
            "Wall Collisions": f"{agg.avg_wall_collisions:.1f}",
            "Achievements": len(agg.unique_achievements_unlocked),
            "Avg Actions/Turn": f"{agg.avg_actions_per_turn_overall:.1f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Task completion breakdown
    print("\n📋 TASK COMPLETION BREAKDOWN")
    print("="*50)
    
    task_data = []
    for agg in mock_results:
        row = {
            "Model": agg.model_name,
            "Difficulty": agg.difficulty,
        }
        for task_type, completion_rate in agg.task_completion_rates.items():
            row[task_type] = f"{completion_rate:.1%}"
        task_data.append(row)
    
    task_df = pd.DataFrame(task_data)
    print(task_df.to_string(index=False))
    
    # Achievement unlock rates
    print("\n🏆 ACHIEVEMENT UNLOCK RATES")
    print("="*50)
    
    achievement_data = []
    for agg in mock_results:
        row = {
            "Model": agg.model_name,
            "Difficulty": agg.difficulty,
        }
        for achievement, rate in agg.achievement_unlock_rates.items():
            row[achievement] = f"{rate:.1%}"
        achievement_data.append(row)
    
    achievement_df = pd.DataFrame(achievement_data)
    print(achievement_df.to_string(index=False))
    
    # Termination breakdown
    print("\n⚰️ TERMINATION BREAKDOWN")
    print("="*50)
    
    termination_data = []
    for agg in mock_results:
        row = {
            "Model": agg.model_name,
            "Difficulty": agg.difficulty,
        }
        for reason, percentage in agg.termination_breakdown.items():
            row[f"{reason.replace('_', ' ').title()} %"] = f"{percentage:.1%}"
        termination_data.append(row)
    
    termination_df = pd.DataFrame(termination_data)
    print(termination_df.to_string(index=False))
    
    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("="*50)
    
    easy_success = mock_results[0].success_rate
    medium_success = mock_results[1].success_rate
    
    print(f"• Overall Success Rate: {(easy_success + medium_success) / 2:.1%}")
    print(f"• Performance Drop (Easy → Medium): {(easy_success - medium_success) * 100:.1f} percentage points")
    print(f"• Best Single Score: {max(agg.composite_score_best for agg in mock_results):.1f}")
    print(f"• Navigation Quality: {'Good' if mock_results[0].navigation_score_avg > 70 else 'Needs Improvement'}")
    print(f"• Exploration Coverage: {mock_results[0].avg_exploration_coverage:.1%} (Easy), {mock_results[1].avg_exploration_coverage:.1%} (Medium)")
    
    print("\n" + "="*80)
    print("✅ This is what your Gemini 1.5 Flash evaluation would look like!")
    print("Run the actual evaluation to get real results.")
    print("="*80)

if __name__ == "__main__":
    display_results_table() 