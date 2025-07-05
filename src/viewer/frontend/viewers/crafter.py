import reflex as rx
import json
from typing import Dict, Any, List
import base64


def CrafterViewer(trace_data: Dict[str, Any]) -> rx.Component:
    """Crafter-specific trace viewer component."""

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

    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading(f"Crafter Trace - {model_name}", size="4"),
            rx.badge(f"Difficulty: {difficulty}", color_scheme="blue"),
            justify_content="space-between",
            width="100%",
        ),
        # Summary stats
        rx.hstack(
            rx.stat(
                rx.stat_label("Total Reward"),
                rx.stat_number(f"{total_reward:.3f}"),
            ),
            rx.stat(
                rx.stat_label("Total Turns"),
                rx.stat_number(str(num_turns)),
            ),
            spacing="4",
            margin_bottom="1rem",
        ),
        # Turn-by-turn viewer
        rx.tabs(
            rx.tab_list(
                *[rx.tab(f"Turn {i + 1}") for i in range(min(num_turns, 10))],
                rx.cond(
                    num_turns > 10,
                    rx.tab(f"... +{num_turns - 10} more"),
                ),
            ),
            rx.tab_panels(
                *[
                    rx.tab_panel(render_crafter_turn(partitions[i], i))
                    for i in range(min(num_turns, 10))
                ],
            ),
            width="100%",
        ),
        width="100%",
        padding="1rem",
        spacing="4",
    )


def render_crafter_turn(partition: Dict[str, Any], turn_idx: int) -> rx.Component:
    """Render a single Crafter turn."""
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

    return rx.vstack(
        # Actions taken
        rx.cond(
            len(actions) > 0,
            rx.vstack(
                rx.text("Actions:", font_weight="bold"),
                rx.hstack(
                    *[rx.badge(action, color_scheme="green") for action in actions],
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
                rx.badge(f"❤️ Health: {stats.get('health', 0)}", color_scheme="red"),
                rx.badge(f"🍖 Food: {stats.get('food', 0)}", color_scheme="orange"),
                rx.badge(f"💧 Drink: {stats.get('drink', 0)}", color_scheme="blue"),
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
