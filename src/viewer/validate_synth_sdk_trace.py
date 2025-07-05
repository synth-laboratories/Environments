#!/usr/bin/env python3
"""Validation functions for Synth SDK traces."""

import sys

sys.path.append("/Users/joshuapurtell/Documents/GitHub/synth-monorepo/synth-sdk")

from synth_sdk.tracing.abstractions import (
    SystemTrace,
    Event,
    AgentComputeStep,
    EnvironmentComputeStep,
    MessageInputs,
    MessageOutputs,
    ArbitraryInputs,
    ArbitraryOutputs,
)
from typing import Dict, List, Any, Optional
import json


def validate_synth_sdk_trace(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a trace has the proper Synth SDK structure with agent compute steps.

    Returns a validation report with details about what's missing or incorrect.
    """
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "total_events": 0,
            "events_with_agent_steps": 0,
            "events_with_env_steps": 0,
            "events_with_messages": 0,
            "events_with_tool_calls": 0,
            "partitions": 0,
        },
    }

    try:
        # Check top-level structure
        if "trace" not in trace_data:
            if "partition" not in trace_data:
                report["errors"].append("Missing 'trace' or 'partition' at top level")
                report["is_valid"] = False
                return report
            else:
                # Direct partition format
                partitions = trace_data["partition"]
        else:
            # Wrapped format
            if "partition" not in trace_data["trace"]:
                report["errors"].append("Missing 'partition' in trace")
                report["is_valid"] = False
                return report
            partitions = trace_data["trace"]["partition"]

        report["stats"]["partitions"] = len(partitions)

        # Validate each partition
        for partition_idx, partition in enumerate(partitions):
            if "events" not in partition:
                report["errors"].append(f"Partition {partition_idx} missing 'events'")
                report["is_valid"] = False
                continue

            events = partition["events"]
            report["stats"]["total_events"] += len(events)

            # Validate each event
            for event_idx, event in enumerate(events):
                event_path = f"Partition {partition_idx}, Event {event_idx}"

                # Check for agent compute step
                if "agent_compute_step" not in event:
                    report["warnings"].append(
                        f"{event_path}: Missing 'agent_compute_step'"
                    )
                elif event["agent_compute_step"] is None:
                    report["warnings"].append(
                        f"{event_path}: 'agent_compute_step' is None"
                    )
                else:
                    report["stats"]["events_with_agent_steps"] += 1
                    agent_step = event["agent_compute_step"]

                    # Validate agent compute step structure
                    if "compute_input" not in agent_step:
                        report["errors"].append(
                            f"{event_path}: Agent step missing 'compute_input'"
                        )
                        report["is_valid"] = False
                    elif not agent_step["compute_input"]:
                        report["warnings"].append(
                            f"{event_path}: Agent step has empty 'compute_input'"
                        )
                    else:
                        # Check for messages in compute_input
                        has_messages = False
                        for input_item in agent_step["compute_input"]:
                            if (
                                isinstance(input_item, dict)
                                and "messages" in input_item
                            ):
                                has_messages = True
                                messages = input_item["messages"]
                                if messages:
                                    report["stats"]["events_with_messages"] += 1
                                    # Check for tool calls in messages
                                    for msg in messages:
                                        if (
                                            isinstance(msg, dict)
                                            and "tool_calls" in msg
                                        ):
                                            report["stats"][
                                                "events_with_tool_calls"
                                            ] += 1
                                            break
                                break

                        if not has_messages:
                            report["warnings"].append(
                                f"{event_path}: Agent step has no messages in compute_input"
                            )

                    if "compute_output" not in agent_step:
                        report["errors"].append(
                            f"{event_path}: Agent step missing 'compute_output'"
                        )
                        report["is_valid"] = False
                    elif not agent_step["compute_output"]:
                        report["warnings"].append(
                            f"{event_path}: Agent step has empty 'compute_output'"
                        )

                    if "model_name" not in agent_step:
                        report["warnings"].append(
                            f"{event_path}: Agent step missing 'model_name'"
                        )

                # Check for environment compute steps
                if "environment_compute_steps" not in event:
                    report["warnings"].append(
                        f"{event_path}: Missing 'environment_compute_steps'"
                    )
                elif event["environment_compute_steps"]:
                    report["stats"]["events_with_env_steps"] += 1

                    # Validate environment steps
                    for env_step_idx, env_step in enumerate(
                        event["environment_compute_steps"]
                    ):
                        if "compute_output" not in env_step:
                            report["errors"].append(
                                f"{event_path}, EnvStep {env_step_idx}: Missing 'compute_output'"
                            )
                            report["is_valid"] = False

    except Exception as e:
        report["errors"].append(f"Exception during validation: {str(e)}")
        report["is_valid"] = False

    return report


def print_validation_report(report: Dict[str, Any], trace_name: str = ""):
    """Print a formatted validation report."""
    print(f"\n🔍 Validation Report{' for ' + trace_name if trace_name else ''}")
    print("=" * 60)

    # Overall status
    status = "✅ VALID" if report["is_valid"] else "❌ INVALID"
    print(f"Status: {status}")

    # Stats
    stats = report["stats"]
    print(f"\n📊 Statistics:")
    print(f"  Partitions: {stats['partitions']}")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  Events with Agent Steps: {stats['events_with_agent_steps']}")
    print(f"  Events with Environment Steps: {stats['events_with_env_steps']}")
    print(f"  Events with Messages: {stats['events_with_messages']}")
    print(f"  Events with Tool Calls: {stats['events_with_tool_calls']}")

    # Errors
    if report["errors"]:
        print(f"\n❌ Errors ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"  - {error}")

    # Warnings
    if report["warnings"]:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if not report["errors"] and not report["warnings"]:
        print("\n✅ No issues found!")


def assert_valid_synth_sdk_trace(trace_data: Dict[str, Any], trace_name: str = ""):
    """
    Assert that a trace has valid Synth SDK structure with agent compute steps.
    Raises AssertionError if validation fails.
    """
    report = validate_synth_sdk_trace(trace_data)

    if not report["is_valid"]:
        error_msg = (
            f"Invalid Synth SDK trace{' (' + trace_name + ')' if trace_name else ''}:\n"
        )
        for error in report["errors"]:
            error_msg += f"  - {error}\n"
        raise AssertionError(error_msg)

    # Also check for critical warnings
    critical_warnings = [
        w
        for w in report["warnings"]
        if "Missing 'agent_compute_step'" in w or "agent_compute_step' is None" in w
    ]

    if critical_warnings:
        warning_msg = f"Synth SDK trace{' (' + trace_name + ')' if trace_name else ''} missing agent compute steps:\n"
        for warning in critical_warnings:
            warning_msg += f"  - {warning}\n"
        raise AssertionError(warning_msg)

    print(
        f"✅ Synth SDK trace{' (' + trace_name + ')' if trace_name else ''} validation passed!"
    )


def assert_trace_has_agent_reasoning(trace_data: Dict[str, Any], trace_name: str = ""):
    """
    Assert that a trace contains actual agent reasoning (messages, tool calls, etc.).
    This is a stricter check than just structural validation.
    """
    report = validate_synth_sdk_trace(trace_data)

    if report["stats"]["events_with_agent_steps"] == 0:
        raise AssertionError(
            f"Trace{' (' + trace_name + ')' if trace_name else ''} has no agent compute steps!"
        )

    if report["stats"]["events_with_messages"] == 0:
        raise AssertionError(
            f"Trace{' (' + trace_name + ')' if trace_name else ''} has no messages in agent compute steps!"
        )

    # Check that we have a reasonable ratio of agent steps to total events
    agent_ratio = (
        report["stats"]["events_with_agent_steps"] / report["stats"]["total_events"]
    )
    if agent_ratio < 0.5:
        raise AssertionError(
            f"Trace{' (' + trace_name + ')' if trace_name else ''} has too few agent compute steps "
            f"({report['stats']['events_with_agent_steps']}/{report['stats']['total_events']} = {agent_ratio:.1%})"
        )

    print(
        f"✅ Trace{' (' + trace_name + ')' if trace_name else ''} has proper agent reasoning!"
    )


if __name__ == "__main__":
    # Test validation on existing traces
    sys.path.append("/Users/joshuapurtell/Documents/GitHub/Environments/src")
    from streamlit_app import (
        get_db_connection,
        load_environments,
        load_evaluations,
        load_trajectories,
        load_trace_data,
    )

    conn = get_db_connection()

    # Test Sokoban traces
    print("Testing Sokoban traces...")
    sokoban_env = [env for env in load_environments(conn) if env["name"] == "sokoban"][
        0
    ]
    sokoban_evals = load_evaluations(conn, sokoban_env["id"])

    if sokoban_evals:
        sokoban_trajectories = load_trajectories(conn, sokoban_evals[0]["id"])
        if sokoban_trajectories:
            trace_data = load_trace_data(conn, sokoban_trajectories[0]["id"])
            if trace_data:
                report = validate_synth_sdk_trace(trace_data)
                print_validation_report(report, "Sokoban")

                # Try assertions
                try:
                    assert_valid_synth_sdk_trace(trace_data, "Sokoban")
                except AssertionError as e:
                    print(f"❌ Assertion failed: {e}")

                try:
                    assert_trace_has_agent_reasoning(trace_data, "Sokoban")
                except AssertionError as e:
                    print(f"❌ Agent reasoning assertion failed: {e}")
