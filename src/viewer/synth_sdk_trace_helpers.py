#!/usr/bin/env python3
"""
Helper functions for creating proper Synth SDK traces in evaluations.
Use these functions to ensure your evaluations produce complete traces
with both agent compute steps and environment compute steps.
"""

import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add synth-sdk to path
sys.path.append("/Users/joshuapurtell/Documents/GitHub/synth-monorepo/synth-sdk")

from synth_sdk.tracing.abstractions import (
    SystemTrace,
    EventPartitionElement,
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
    MessageInputs,
    MessageOutputs,
    ArbitraryInputs,
    ArbitraryOutputs,
)
from validate_synth_sdk_trace import (
    validate_synth_sdk_trace,
    assert_trace_has_agent_reasoning,
)


def create_agent_compute_step(
    model_name: str,
    input_messages: List[Dict[str, Any]],
    output_messages: List[Dict[str, Any]],
    model_params: Optional[Dict[str, Any]] = None,
    event_order: int = 1,
) -> Dict[str, Any]:
    """
    Create a proper agent compute step with messages.

    Args:
        model_name: The model used (e.g., "gpt-4o-mini")
        input_messages: List of messages sent to the agent (system, user, etc.)
        output_messages: List of messages from the agent (assistant, tool results)
        model_params: Optional model parameters (temperature, max_tokens, etc.)
        event_order: Order of this compute step in the event

    Returns:
        Dict representing the agent compute step
    """

    if model_params is None:
        model_params = {"temperature": 0.7, "max_tokens": 1000}

    return {
        "event_order": event_order,
        "compute_began": datetime.now().isoformat(),
        "compute_ended": datetime.now().isoformat(),
        "model_name": model_name,
        "model_params": model_params,
        "compute_input": [{"messages": input_messages}],
        "compute_output": [{"messages": output_messages}],
    }


def create_environment_compute_step(
    action_input: Dict[str, Any],
    environment_output: Dict[str, Any],
    event_order: int = 2,
) -> Dict[str, Any]:
    """
    Create a proper environment compute step.

    Args:
        action_input: The action sent to the environment
        environment_output: The environment's response (state, reward, etc.)
        event_order: Order of this compute step in the event

    Returns:
        Dict representing the environment compute step
    """

    return {
        "event_order": event_order,
        "compute_began": datetime.now().isoformat(),
        "compute_ended": datetime.now().isoformat(),
        "compute_input": [{"inputs": action_input}],
        "compute_output": [{"outputs": environment_output}],
    }


def create_synth_sdk_event(
    event_type: str,
    partition_index: int,
    agent_compute_step: Dict[str, Any],
    environment_compute_steps: List[Dict[str, Any]],
    event_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a proper Synth SDK event with both agent and environment compute steps.

    Args:
        event_type: Type of event (e.g., "sokoban_turn")
        partition_index: Index of this partition
        agent_compute_step: The agent's reasoning step
        environment_compute_steps: List of environment interaction steps
        event_metadata: Optional metadata for this event

    Returns:
        Dict representing the complete event
    """

    if event_metadata is None:
        event_metadata = {}

    return {
        "event_type": event_type,
        "opened": datetime.now().timestamp(),
        "closed": datetime.now().timestamp() + 1,
        "partition_index": partition_index,
        "event_metadata": event_metadata,
        "agent_compute_step": agent_compute_step,
        "environment_compute_steps": environment_compute_steps,
    }


def create_synth_sdk_trace(
    system_name: str,
    system_id: str,
    system_instance_id: str,
    partitions: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    reward_signals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a complete Synth SDK trace with validation.

    Args:
        system_name: Name of the system (e.g., "sokoban_agent")
        system_id: ID of the system (e.g., "sokoban-v1")
        system_instance_id: Unique instance ID
        partitions: List of partition data with events
        metadata: Optional system metadata
        questions: Optional training questions
        reward_signals: Optional reward signals

    Returns:
        Complete Synth SDK trace dict
    """

    if metadata is None:
        metadata = {}

    if questions is None:
        questions = []

    if reward_signals is None:
        reward_signals = []

    trace_data = {
        "trace": {
            "system_name": system_name,
            "system_id": system_id,
            "system_instance_id": system_instance_id,
            "metadata": metadata,
            "partition": partitions,
        },
        "dataset": {"questions": questions, "reward_signals": reward_signals},
    }

    return trace_data


def validate_and_save_trace(
    trace_data: Dict[str, Any], output_path: Path, trace_name: str = ""
) -> bool:
    """
    Validate a trace and save it if valid.

    Args:
        trace_data: The trace data to validate
        output_path: Path to save the trace
        trace_name: Optional name for error messages

    Returns:
        True if valid and saved, False otherwise
    """

    try:
        # Validate the trace
        assert_trace_has_agent_reasoning(trace_data, trace_name)

        # Save the trace
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(trace_data, f, indent=2)

        print(f"✅ Saved valid trace to: {output_path}")
        return True

    except AssertionError as e:
        print(f"❌ Trace validation failed for {trace_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error saving trace {trace_name}: {e}")
        return False


def create_sokoban_turn_with_reasoning(
    turn_number: int,
    partition_index: int,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    agent_reasoning: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    action_input: Dict[str, Any],
    environment_output: Dict[str, Any],
    turn_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a complete Sokoban turn with agent reasoning and environment interaction.

    This is a high-level helper that creates a complete event with both
    agent compute step and environment compute step.

    Args:
        turn_number: Turn number (1-indexed)
        partition_index: Partition index (0-indexed)
        model_name: Model used for reasoning
        system_prompt: System message content
        user_prompt: User message content (game state)
        agent_reasoning: Agent's reasoning/thinking
        tool_calls: List of tool calls made by agent
        tool_results: List of tool results
        action_input: Action sent to environment
        environment_output: Environment's response
        turn_metadata: Optional metadata for this turn

    Returns:
        Complete partition dict with event
    """

    # Create input messages
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Create output messages
    output_messages = [
        {"role": "assistant", "content": agent_reasoning, "tool_calls": tool_calls}
    ]

    # Add tool results
    for tool_result in tool_results:
        output_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_result["tool_call_id"],
                "content": tool_result["content"],
            }
        )

    # Create agent compute step
    agent_step = create_agent_compute_step(
        model_name=model_name,
        input_messages=input_messages,
        output_messages=output_messages,
        event_order=1,
    )

    # Create environment compute step
    env_step = create_environment_compute_step(
        action_input=action_input, environment_output=environment_output, event_order=2
    )

    # Create event
    event = create_synth_sdk_event(
        event_type="sokoban_turn",
        partition_index=partition_index,
        agent_compute_step=agent_step,
        environment_compute_steps=[env_step],
        event_metadata=turn_metadata or {"turn_number": turn_number},
    )

    # Return partition
    return {"partition_index": partition_index, "events": [event]}


# Example usage functions
def example_sokoban_evaluation_with_proper_tracing():
    """
    Example of how to use these helpers in a Sokoban evaluation.
    This shows the pattern you should follow in your evaluations.
    """

    print("📝 Example: How to create proper Synth SDK traces in evaluations")
    print("=" * 70)

    # 1. Set up trace metadata
    system_name = "sokoban_agent"
    system_id = "sokoban-v1"
    system_instance_id = "example-instance-123"

    trace_metadata = {
        "model_name": "gpt-4o-mini",
        "difficulty": "ultra-easy",
        "seed": 42,
        "success": True,
        "final_reward": 0.97,
    }

    # 2. Create partitions for each turn
    partitions = []

    # Turn 1
    partition_1 = create_sokoban_turn_with_reasoning(
        turn_number=1,
        partition_index=0,
        model_name="gpt-4o-mini",
        system_prompt="You are playing Sokoban. Push all boxes (O) onto targets (X).",
        user_prompt="Current state: Player at [3,3], box at [1,2], target at [2,2]",
        agent_reasoning="I need to get behind the box to push it onto the target.",
        tool_calls=[
            {
                "id": "move_1",
                "type": "function",
                "function": {"name": "move", "arguments": '{"direction": "up"}'},
            }
        ],
        tool_results=[{"tool_call_id": "move_1", "content": "Moved up successfully"}],
        action_input={"action": "up", "action_index": 0},
        environment_output={
            "room_text": "# # # #\n# _ _ #\n# O X P #\n# # # #",
            "player_position": [3, 2],
            "reward": 0.0,
            "terminated": False,
        },
    )
    partitions.append(partition_1)

    # Turn 2
    partition_2 = create_sokoban_turn_with_reasoning(
        turn_number=2,
        partition_index=1,
        model_name="gpt-4o-mini",
        system_prompt="You are playing Sokoban. Push all boxes (O) onto targets (X).",
        user_prompt="Current state: Player at [3,2], box at [1,2], target at [2,2]",
        agent_reasoning="Perfect! Now I can push the box onto the target.",
        tool_calls=[
            {
                "id": "move_2",
                "type": "function",
                "function": {"name": "move", "arguments": '{"direction": "left"}'},
            }
        ],
        tool_results=[
            {
                "tool_call_id": "move_2",
                "content": "Pushed box onto target! Puzzle solved!",
            }
        ],
        action_input={"action": "left", "action_index": 2},
        environment_output={
            "room_text": "# # # #\n# _ _ #\n# _ P _ #\n# # # #",
            "player_position": [2, 2],
            "reward": 0.97,
            "terminated": True,
        },
    )
    partitions.append(partition_2)

    # 3. Create complete trace
    trace_data = create_synth_sdk_trace(
        system_name=system_name,
        system_id=system_id,
        system_instance_id=system_instance_id,
        partitions=partitions,
        metadata=trace_metadata,
        questions=[
            {
                "id": "sokoban_puzzle",
                "intent": "solve the puzzle",
                "criteria": "push all boxes onto targets",
            }
        ],
        reward_signals=[
            {
                "question_id": "sokoban_puzzle",
                "system_instance_id": system_instance_id,
                "reward": 0.97,
                "annotation": "Successfully solved puzzle",
            }
        ],
    )

    # 4. Validate and save
    output_path = Path("example_proper_trace.json")
    success = validate_and_save_trace(trace_data, output_path, "Example Sokoban")

    if success:
        print("\n✅ Example trace created successfully!")
        print("\n📋 Key points for your evaluations:")
        print(
            "   1. Always create both agent_compute_step AND environment_compute_steps"
        )
        print("   2. Capture the actual messages sent to/from the agent")
        print("   3. Include tool calls and tool results in the messages")
        print("   4. Use validate_and_save_trace() to ensure correctness")
        print("   5. Add assertions in your evaluation code to catch issues early")

    return success


if __name__ == "__main__":
    example_sokoban_evaluation_with_proper_tracing()
