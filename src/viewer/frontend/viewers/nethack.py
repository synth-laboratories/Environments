import reflex as rx
import json
from typing import Dict, Any, List


def NethackViewer(trace_data: Dict[str, Any]) -> rx.Component:
    """NetHack-specific trace viewer component."""

    # Extract trace structure
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})
    partitions = trace.get("partition", [])

    # Extract summary data
    model_name = metadata.get("model_name", "Unknown")
    total_reward = dataset.get("reward_signals", [{}])[0].get("reward", 0.0)
    num_turns = len(partitions)

    return rx.vstack(
        # Header
        rx.heading(f"NetHack Trace - {model_name}", size="4"),
        # Summary stats
        rx.hstack(
            rx.stat(
                rx.stat_label("Total Score"),
                rx.stat_number(f"{total_reward:.0f}"),
            ),
            rx.stat(
                rx.stat_label("Total Turns"),
                rx.stat_number(str(num_turns)),
            ),
            rx.stat(
                rx.stat_label("Dungeon Level"),
                rx.stat_number(str(metadata.get("final_dlevel", 1))),
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
                    rx.tab_panel(render_nethack_turn(partitions[i], i))
                    for i in range(min(num_turns, 10))
                ],
            ),
            width="100%",
        ),
        width="100%",
        padding="1rem",
        spacing="4",
    )


def render_nethack_turn(partition: Dict[str, Any], turn_idx: int) -> rx.Component:
    """Render a single NetHack turn."""
    events = partition.get("events", [])
    if not events:
        return rx.text("No events in this turn", color="gray.500")

    event = events[0]

    # Extract environment steps
    env_steps = event.get("environment_compute_steps", [])

    # Collect game state
    messages = []
    stats = {}
    glyphs = None
    actions = []

    for step in env_steps:
        outputs = step.get("compute_output", [{}])[0].get("outputs", {})

        # Extract messages
        if "message" in outputs:
            messages.append(outputs["message"])

        # Extract stats
        if "blstats" in outputs:
            stats = parse_nethack_stats(outputs["blstats"])

        # Extract glyphs (ASCII representation)
        if "glyphs" in outputs:
            glyphs = outputs["glyphs"]

        # Extract action
        if "action" in outputs:
            action = outputs["action"]
            if isinstance(action, int):
                action_name = get_nethack_action_name(action)
                actions.append(action_name)
            else:
                actions.append(str(action))

    return rx.vstack(
        # Actions
        rx.cond(
            len(actions) > 0,
            rx.vstack(
                rx.text("Actions:", font_weight="bold"),
                rx.hstack(
                    *[rx.badge(action, color_scheme="purple") for action in actions],
                    wrap="wrap",
                    spacing="2",
                ),
                spacing="2",
            ),
        ),
        # Stats
        rx.cond(
            len(stats) > 0,
            rx.hstack(
                rx.text("Stats:", font_weight="bold"),
                rx.badge(
                    f"HP: {stats.get('hp', 0)}/{stats.get('max_hp', 0)}",
                    color_scheme="red",
                ),
                rx.badge(f"Level: {stats.get('level', 1)}", color_scheme="blue"),
                rx.badge(f"Gold: {stats.get('gold', 0)}", color_scheme="yellow"),
                rx.badge(f"AC: {stats.get('ac', 10)}", color_scheme="green"),
                spacing="2",
            ),
        ),
        # Messages
        rx.cond(
            len(messages) > 0,
            rx.vstack(
                rx.text("Messages:", font_weight="bold"),
                rx.vstack(
                    *[
                        rx.text(msg, font_family="monospace", font_size="0.9rem")
                        for msg in messages[-5:]  # Show last 5 messages
                    ],
                    spacing="1",
                    padding="0.5rem",
                    bg="gray.100",
                    border_radius="4px",
                ),
                spacing="2",
            ),
        ),
        # ASCII display (if available)
        rx.cond(
            glyphs is not None,
            rx.vstack(
                rx.text("Game View:", font_weight="bold"),
                rx.box(
                    rx.text(
                        format_nethack_glyphs(glyphs),
                        font_family="monospace",
                        font_size="0.8rem",
                        white_space="pre",
                    ),
                    padding="0.5rem",
                    bg="black",
                    color="green.400",
                    border_radius="4px",
                    overflow_x="auto",
                ),
                spacing="2",
            ),
        ),
        spacing="4",
        padding="1rem",
        border="1px solid #eee",
        border_radius="8px",
    )


def parse_nethack_stats(blstats: List[int]) -> Dict[str, int]:
    """Parse NetHack blstats array into readable stats."""
    # NetHack blstats indices (simplified)
    stats = {}
    if len(blstats) >= 25:
        stats["hp"] = blstats[10]
        stats["max_hp"] = blstats[11]
        stats["level"] = blstats[18]
        stats["gold"] = blstats[13]
        stats["ac"] = blstats[17]
    return stats


def format_nethack_glyphs(glyphs: List[List[int]]) -> str:
    """Format NetHack glyphs array into ASCII representation."""
    # This is a simplified version - real implementation would map glyph IDs to ASCII
    if not glyphs:
        return "No map data available"

    # For now, just show dimensions
    return f"Map: {len(glyphs)}x{len(glyphs[0]) if glyphs else 0} (glyph data not rendered)"


def get_nethack_action_name(action_idx: int) -> str:
    """Map NetHack action index to name (simplified)."""
    # This is a subset of NetHack actions
    action_names = {
        0: "move_nw",
        1: "move_n",
        2: "move_ne",
        3: "move_w",
        4: "wait",
        5: "move_e",
        6: "move_sw",
        7: "move_s",
        8: "move_se",
        13: "kick",
        20: "search",
        21: "open",
        31: "pickup",
        # ... many more actions
    }
    return action_names.get(action_idx, f"action_{action_idx}")
