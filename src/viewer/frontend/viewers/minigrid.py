import reflex as rx
import json
from typing import Dict, Any, List
import base64


def MinigridViewer(trace_data: Dict[str, Any]) -> rx.Component:
    """MiniGrid-specific trace viewer component."""

    # Extract trace structure
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})
    partitions = trace.get("partition", [])

    # Extract summary data
    model_name = metadata.get("model_name", "Unknown")
    env_name = metadata.get("env_name", "Unknown MiniGrid Environment")
    total_reward = dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
    num_turns = len(partitions)
    success = metadata.get("success", False)

    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading(f"MiniGrid - {env_name}", size="4"),
            rx.badge(f"Model: {model_name}", color_scheme="purple"),
            rx.cond(
                success,
                rx.badge("Success", color_scheme="green"),
                rx.badge("Failed", color_scheme="red"),
            ),
            justify_content="space-between",
            width="100%",
        ),
        # Summary stats
        rx.hstack(
            rx.stat(
                rx.stat_label("Total Reward"),
                rx.stat_number(f"{total_reward:.2f}"),
            ),
            rx.stat(
                rx.stat_label("Total Steps"),
                rx.stat_number(str(num_turns)),
            ),
            rx.stat(
                rx.stat_label("Final Status"),
                rx.stat_number("✓" if success else "✗"),
            ),
            spacing="4",
            margin_bottom="1rem",
        ),
        # Turn-by-turn viewer
        rx.tabs(
            rx.tab_list(
                *[rx.tab(f"Step {i + 1}") for i in range(min(num_turns, 15))],
                rx.cond(
                    num_turns > 15,
                    rx.tab(f"... +{num_turns - 15} more"),
                ),
            ),
            rx.tab_panels(
                *[
                    rx.tab_panel(render_minigrid_turn(partitions[i], i))
                    for i in range(min(num_turns, 15))
                ],
            ),
            width="100%",
        ),
        width="100%",
        padding="1rem",
        spacing="4",
    )


def render_minigrid_turn(partition: Dict[str, Any], turn_idx: int) -> rx.Component:
    """Render a single MiniGrid turn."""
    events = partition.get("events", [])
    if not events:
        return rx.text("No events in this turn", color="gray.500")

    event = events[0]

    # Extract environment steps
    env_steps = event.get("environment_compute_steps", [])

    # Collect observations and actions
    observations = []
    actions = []
    rewards = []
    mission = ""
    grid_repr = None

    for step in env_steps:
        outputs = step.get("compute_output", [{}])[0].get("outputs", {})

        # Extract observation (image or grid)
        if "observation" in outputs:
            obs = outputs["observation"]
            if isinstance(obs, dict):
                # Extract image if available
                if "image" in obs:
                    img_data = obs["image"]
                    if isinstance(img_data, str):
                        if not img_data.startswith("data:"):
                            img_data = f"data:image/png;base64,{img_data}"
                        observations.append(img_data)
                    elif "image_base64" in obs:
                        observations.append(
                            f"data:image/png;base64,{obs['image_base64']}"
                        )

                # Extract mission
                if "mission" in obs:
                    mission = obs["mission"]

        # Extract image directly
        if "image_base64" in outputs:
            observations.append(f"data:image/png;base64,{outputs['image_base64']}")
        elif "image" in outputs and isinstance(outputs["image"], str):
            img_data = outputs["image"]
            if not img_data.startswith("data:"):
                img_data = f"data:image/png;base64,{img_data}"
            observations.append(img_data)

        # Extract grid representation
        if "grid" in outputs:
            grid_repr = outputs["grid"]

        # Extract action
        if "action" in outputs:
            action = outputs["action"]
            if isinstance(action, int):
                action_name = get_minigrid_action_name(action)
                actions.append(action_name)
            else:
                actions.append(str(action))

        # Extract reward
        if "reward" in outputs:
            rewards.append(float(outputs["reward"]))

    return rx.vstack(
        # Mission
        rx.cond(
            mission != "",
            rx.vstack(
                rx.text("Mission:", font_weight="bold"),
                rx.text(mission, font_style="italic", color="gray.700"),
                spacing="1",
            ),
        ),
        # Actions taken
        rx.cond(
            len(actions) > 0,
            rx.vstack(
                rx.text("Action:", font_weight="bold"),
                rx.hstack(
                    *[rx.badge(action, color_scheme="blue") for action in actions],
                    wrap="wrap",
                    spacing="2",
                ),
                spacing="2",
            ),
        ),
        # Rewards
        rx.cond(
            len(rewards) > 0 and any(r != 0 for r in rewards),
            rx.hstack(
                rx.text("Reward:", font_weight="bold"),
                *[
                    rx.badge(f"{r:+.2f}", color_scheme="green" if r > 0 else "red")
                    for r in rewards
                    if r != 0
                ],
                spacing="2",
            ),
        ),
        # Grid visualization
        rx.cond(
            len(observations) > 0,
            rx.vstack(
                rx.text("Grid State:", font_weight="bold"),
                rx.hstack(
                    *[
                        rx.image(
                            src=obs,
                            width="300px",
                            height="300px",
                            object_fit="contain",
                            border="2px solid #ddd",
                            border_radius="4px",
                            image_rendering="pixelated",  # For crisp pixel art
                        )
                        for obs in observations[:2]  # Show max 2 images
                    ],
                    spacing="2",
                ),
                spacing="2",
            ),
            rx.cond(
                grid_repr is not None,
                rx.vstack(
                    rx.text("Grid State (ASCII):", font_weight="bold"),
                    rx.box(
                        rx.text(
                            format_minigrid_ascii(grid_repr),
                            font_family="monospace",
                            font_size="0.9rem",
                            white_space="pre",
                        ),
                        padding="0.5rem",
                        bg="gray.100",
                        border_radius="4px",
                    ),
                    spacing="2",
                ),
                rx.text(
                    "No visualization available for this step",
                    color="gray.500",
                    font_style="italic",
                ),
            ),
        ),
        spacing="4",
        padding="1rem",
        border="1px solid #eee",
        border_radius="8px",
    )


def get_minigrid_action_name(action_idx: int) -> str:
    """Map MiniGrid action index to name."""
    action_names = {
        0: "left",
        1: "right",
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done",
    }
    return action_names.get(action_idx, f"action_{action_idx}")


def format_minigrid_ascii(grid_data) -> str:
    """Format MiniGrid grid data as ASCII representation."""
    if isinstance(grid_data, str):
        return grid_data
    elif isinstance(grid_data, list):
        # Convert grid array to ASCII
        return "\n".join(["".join(row) for row in grid_data])
    else:
        return f"Grid data format not supported: {type(grid_data)}"
