import reflex as rx
import json
from typing import Dict, Any, List
import base64


def CrafterViewerCompat(trace_data: Dict[str, Any]) -> rx.Component:
    """Crafter-specific trace viewer component compatible with Reflex 0.3.8."""

    # Extract trace structure
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})
    partitions = trace.get("partition", [])

    # Extract summary data
    model_name = metadata.get("model_name", "Unknown")
    difficulty = metadata.get("difficulty", "Unknown")
    total_reward = dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
    num_turns = len(partitions)

    # For Reflex 0.3.8, we'll show first few turns directly
    turn_components = []
    for i in range(min(num_turns, 5)):  # Show first 5 turns
        turn_components.append(
            rx.vstack(
                rx.heading(f"Turn {i + 1}", size="5"),
                render_crafter_turn_compat(partitions[i], i),
                width="100%",
                margin_bottom="1rem",
            )
        )

    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading(f"Crafter Trace - {model_name}", size="4"),
            rx.text(
                f"Difficulty: {difficulty}",
                padding="0.25rem 0.5rem",
                bg="blue.100",
                color="blue.800",
                border_radius="4px",
                font_size="0.9rem",
            ),
            justify_content="space-between",
            width="100%",
        ),
        # Summary stats
        rx.hstack(
            rx.vstack(
                rx.text("Total Reward", font_size="0.8rem", color="gray.600"),
                rx.text(f"{total_reward:.3f}", font_size="1.5rem", font_weight="bold"),
                align_items="center",
                bg="gray.50",
                padding="1rem",
                border_radius="8px",
            ),
            rx.vstack(
                rx.text("Total Turns", font_size="0.8rem", color="gray.600"),
                rx.text(str(num_turns), font_size="1.5rem", font_weight="bold"),
                align_items="center",
                bg="gray.50",
                padding="1rem",
                border_radius="8px",
            ),
            spacing="4",
            margin_bottom="1rem",
        ),
        # Turn details
        rx.vstack(
            rx.heading("Turn Details", size="4"),
            *turn_components,
            rx.cond(
                num_turns > 5,
                rx.text(
                    f"... and {num_turns - 5} more turns",
                    color="gray.500",
                    font_style="italic",
                ),
            ),
            width="100%",
        ),
        width="100%",
        padding="1rem",
        spacing="4",
    )


def render_crafter_turn_compat(
    partition: Dict[str, Any], turn_idx: int
) -> rx.Component:
    """Render a single Crafter turn for Reflex 0.3.8."""
    events = partition.get("events", [])
    if not events:
        return rx.text("No events in this turn", color="gray.500")

    event = events[0]  # Usually one event per turn

    # Extract environment steps
    env_steps = event.get("environment_compute_steps", [])

    # Collect images and actions
    images = []
    actions = []
    stats = {}

    for step in env_steps:
        outputs = step.get("compute_output", [{}])[0].get("outputs", {})

        # Extract image
        if "image_base64" in outputs:
            images.append(outputs["image_base64"])
        elif "image" in outputs:
            img_data = outputs["image"]
            if not img_data.startswith("data:"):
                img_data = f"data:image/png;base64,{img_data}"
            images.append(img_data)

        # Extract action
        action_idx = outputs.get("action_index", -1)
        if action_idx >= 0:
            action_name = get_crafter_action_name(action_idx)
            actions.append(f"{action_name} ({action_idx})")

        # Extract stats
        if "player_stats" in outputs:
            stats = outputs["player_stats"]

    # Create action badges
    action_components = []
    for action in actions:
        action_components.append(
            rx.text(
                action,
                padding="0.25rem 0.5rem",
                bg="green.100",
                color="green.800",
                border_radius="4px",
                font_size="0.9rem",
                margin_right="0.5rem",
            )
        )

    return rx.vstack(
        # Actions taken
        rx.cond(
            len(actions) > 0,
            rx.vstack(
                rx.text("Actions:", font_weight="bold"),
                rx.hstack(
                    *action_components,
                    wrap="wrap",
                    spacing="2",
                ),
                spacing="2",
            ),
        ),
        # Player stats
        rx.cond(
            len(stats) > 0,
            rx.hstack(
                rx.text("Stats:", font_weight="bold"),
                rx.text(
                    f"❤️ Health: {stats.get('health', 0)}",
                    padding="0.25rem 0.5rem",
                    bg="red.100",
                    color="red.800",
                    border_radius="4px",
                    font_size="0.9rem",
                ),
                rx.text(
                    f"🍖 Food: {stats.get('food', 0)}",
                    padding="0.25rem 0.5rem",
                    bg="orange.100",
                    color="orange.800",
                    border_radius="4px",
                    font_size="0.9rem",
                ),
                rx.text(
                    f"💧 Drink: {stats.get('drink', 0)}",
                    padding="0.25rem 0.5rem",
                    bg="blue.100",
                    color="blue.800",
                    border_radius="4px",
                    font_size="0.9rem",
                ),
                spacing="2",
            ),
        ),
        # Images
        rx.cond(
            len(images) > 0,
            rx.vstack(
                rx.text("Game State:", font_weight="bold"),
                rx.hstack(
                    *[
                        rx.image(
                            src=img
                            if img.startswith("data:")
                            else f"data:image/png;base64,{img}",
                            width="200px",
                            height="200px",
                            object_fit="contain",
                            border="1px solid #ddd",
                            border_radius="4px",
                        )
                        for img in images[:3]  # Show max 3 images
                    ],
                    spacing="2",
                ),
                spacing="2",
            ),
            rx.text(
                "No images available for this turn",
                color="gray.500",
                font_style="italic",
            ),
        ),
        spacing="4",
        padding="1rem",
        border="1px solid #eee",
        border_radius="8px",
    )


def get_crafter_action_name(action_idx: int) -> str:
    """Map Crafter action index to name."""
    action_names = {
        0: "noop",
        1: "move_left",
        2: "move_right",
        3: "move_up",
        4: "move_down",
        5: "do",
        6: "sleep",
        7: "place_stone",
        8: "place_table",
        9: "place_furnace",
        10: "place_plant",
        11: "make_wood_pickaxe",
        12: "make_stone_pickaxe",
        13: "make_iron_pickaxe",
        14: "make_wood_sword",
        15: "make_stone_sword",
        16: "make_iron_sword",
    }
    return action_names.get(action_idx, f"unknown_{action_idx}")
