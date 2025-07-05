import reflex as rx
import json
from typing import Dict, List, Optional, Any, Union
import sys
from pathlib import Path

# Import database functions from local module
from .database import (
    list_evaluations,
    list_traces,
    get_trace,
    get_trace_from_file,
    get_environments,
)


# Environment-specific trace processing
def process_trace_for_environment(env_name: str, trace_data: Dict) -> Dict[str, Any]:
    """Process trace data for a specific environment."""
    # Extract basic metadata
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})

    model_name = metadata.get("model_name", "Unknown")
    difficulty = metadata.get("difficulty", "Unknown")

    # Extract reward
    reward_signals = dataset.get("reward_signals", [{}])
    total_reward = reward_signals[0].get("reward", 0.0) if reward_signals else 0.0

    # Process turns based on environment
    partitions = trace.get("partition", [])
    num_turns = len(partitions)

    if env_name == "crafter":
        turns = _process_crafter_turns(partitions)
    elif env_name == "nethack":
        turns = _process_nethack_turns(partitions)
    elif env_name == "minigrid":
        turns = _process_minigrid_turns(partitions)
    else:
        turns = _process_generic_turns(partitions)

    return {
        "model_name": model_name,
        "difficulty": difficulty,
        "total_reward": total_reward,
        "num_turns": num_turns,
        "turns": turns,
    }


def _process_crafter_turns(partitions: List[Dict]) -> List[Dict]:
    """Process Crafter-specific turn data."""
    turns = []
    for i, partition in enumerate(partitions[:10]):  # Limit for performance
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        env_steps = event.get("environment_compute_steps", [])

        # Collect turn data
        images = []
        actions = []
        stats = None
        achievements = event.get("event_metadata", {}).get("new_achievements", [])

        for step_idx, step in enumerate(env_steps):
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})

            # Extract image
            if "image_base64" in outputs:
                img_data = outputs["image_base64"]
                if not img_data.startswith("data:"):
                    img_data = f"data:image/png;base64,{img_data}"

                action_idx = outputs.get("action_index", -1)
                action_name = _get_crafter_action_name(action_idx)

                images.append(
                    {
                        "data_url": img_data,
                        "caption": f"{step_idx}. {action_name}",
                        "step_number": step_idx,
                    }
                )

            # Extract action
            action_idx = outputs.get("action_index", -1)
            if action_idx >= 0:
                action_name = _get_crafter_action_name(action_idx)
                actions.append(
                    {
                        "name": action_name,
                        "index": action_idx,
                        "display_name": f"{action_name} ({action_idx})",
                    }
                )

            # Extract stats
            if "player_stats" in outputs:
                player_stats = outputs["player_stats"]
                stats = {
                    "health": player_stats.get("health", 0),
                    "food": player_stats.get("food", 0),
                    "drink": player_stats.get("drink", 0),
                }

        turns.append(
            {
                "turn_number": i + 1,
                "images": images,
                "actions": actions,
                "stats": stats,
                "achievements": achievements,
                "metadata": {},
            }
        )

    return turns


def _process_nethack_turns(partitions: List[Dict]) -> List[Dict]:
    """Process NetHack-specific turn data."""
    turns = []
    for i, partition in enumerate(partitions[:10]):
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        env_steps = event.get("environment_compute_steps", [])

        # Collect NetHack-specific data
        messages = []
        actions = []
        stats = None

        for step in env_steps:
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})

            # Extract messages
            if "message" in outputs:
                messages.append(outputs["message"])

            # Extract stats
            if "blstats" in outputs and len(outputs["blstats"]) >= 25:
                blstats = outputs["blstats"]
                stats = {
                    "hp": blstats[10],
                    "max_hp": blstats[11],
                    "level": blstats[18],
                    "gold": blstats[13],
                    "ac": blstats[17],
                }

            # Extract action
            if "action" in outputs:
                action = outputs["action"]
                if isinstance(action, int):
                    action_name = f"action_{action}"
                    actions.append(
                        {
                            "name": action_name,
                            "index": action,
                            "display_name": action_name,
                        }
                    )

        turns.append(
            {
                "turn_number": i + 1,
                "images": [],  # NetHack uses ASCII
                "actions": actions,
                "stats": stats,
                "achievements": [],  # NetHack doesn't have achievements
                "metadata": {
                    "messages": messages[-5:] if messages else [],  # Last 5 messages
                },
            }
        )

    return turns


def _process_minigrid_turns(partitions: List[Dict]) -> List[Dict]:
    """Process MiniGrid-specific turn data."""
    turns = []
    for i, partition in enumerate(partitions[:15]):  # MiniGrid episodes can be short
        events = partition.get("events", [])
        if not events:
            continue

        event = events[0]
        env_steps = event.get("environment_compute_steps", [])

        # Collect MiniGrid-specific data
        images = []
        actions = []
        rewards = []
        mission = ""

        for step in env_steps:
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})

            # Extract observation
            if "observation" in outputs:
                obs = outputs["observation"]
                if isinstance(obs, dict):
                    # Extract image if available
                    if "image" in obs:
                        img_data = obs["image"]
                        if isinstance(img_data, str):
                            if not img_data.startswith("data:"):
                                img_data = f"data:image/png;base64,{img_data}"
                            images.append(
                                {
                                    "data_url": img_data,
                                    "caption": f"Step {i}",
                                    "step_number": i,
                                }
                            )
                    elif "image_base64" in obs:
                        images.append(
                            {
                                "data_url": f"data:image/png;base64,{obs['image_base64']}",
                                "caption": f"Step {i}",
                                "step_number": i,
                            }
                        )

                    # Extract mission
                    if "mission" in obs:
                        mission = obs["mission"]

            # Extract image directly
            if "image_base64" in outputs:
                images.append(
                    {
                        "data_url": f"data:image/png;base64,{outputs['image_base64']}",
                        "caption": f"Step {i}",
                        "step_number": i,
                    }
                )
            elif "image" in outputs and isinstance(outputs["image"], str):
                img_data = outputs["image"]
                if not img_data.startswith("data:"):
                    img_data = f"data:image/png;base64,{img_data}"
                images.append(
                    {"data_url": img_data, "caption": f"Step {i}", "step_number": i}
                )

            # Extract action
            if "action" in outputs:
                action = outputs["action"]
                if isinstance(action, int):
                    action_name = _get_minigrid_action_name(action)
                    actions.append(
                        {
                            "name": action_name,
                            "index": action,
                            "display_name": action_name,
                        }
                    )

            # Extract reward
            if "reward" in outputs:
                rewards.append(float(outputs["reward"]))

        turns.append(
            {
                "turn_number": i + 1,
                "images": images,
                "actions": actions,
                "stats": None,  # MiniGrid doesn't have player stats
                "achievements": [],  # MiniGrid doesn't have achievements
                "metadata": {"mission": mission, "rewards": rewards},
            }
        )

    return turns


def _process_generic_turns(partitions: List[Dict]) -> List[Dict]:
    """Process generic turn data."""
    turns = []
    for i, partition in enumerate(partitions[:5]):  # Fewer turns for generic
        turns.append(
            {
                "turn_number": i + 1,
                "images": [],
                "actions": [],
                "stats": None,
                "achievements": [],
                "metadata": {},
            }
        )
    return turns


def _get_crafter_action_name(action_idx: int) -> str:
    """Map Crafter action index to name."""
    action_names = {
        -1: "initial state",
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


def _get_minigrid_action_name(action_idx: int) -> str:
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


class State(rx.State):
    """Main application state."""

    # Data
    environments: List[str] = []
    evaluations: List[Dict] = []
    traces: List[Dict] = []
    selected_env: Optional[str] = None
    selected_run: Optional[Dict] = None
    selected_trace: Optional[Dict] = None
    current_trace_data: Optional[Dict] = None

    # UI State
    loading: bool = False
    error: Optional[str] = None
    show_raw_data: bool = False

    # Processed trace data for visualization (typed!)
    trace_model_name: str = ""
    trace_difficulty: str = ""
    trace_total_reward: float = 0.0
    trace_num_turns: int = 0
    trace_turns: List[Dict[str, Any]] = []  # Will contain serialized TraceTurn data

    def load_environments(self):
        """Load available environments."""
        self.loading = True
        self.error = None
        try:
            self.environments = get_environments()
            if not self.environments:
                self.error = "No environments found"
        except Exception as e:
            self.error = f"Failed to load environments: {str(e)}"
        finally:
            self.loading = False

    def select_environment(self, env_name: str):
        """Select an environment and load its evaluations."""
        self.selected_env = env_name
        self.selected_run = None
        self.selected_trace = None
        self.current_trace_data = None
        self._clear_processed_trace_data()
        self.load_evaluations()

    def load_evaluations(self):
        """Load evaluations for selected environment."""
        if not self.selected_env:
            return

        self.loading = True
        self.error = None
        try:
            df = list_evaluations(self.selected_env)
            self.evaluations = df.to_dict("records")
        except Exception as e:
            self.error = f"Failed to load evaluations: {str(e)}"
            self.evaluations = []
        finally:
            self.loading = False

    def select_run(self, run_id: str):
        """Select a run and load its traces."""
        self.selected_run = next(
            (e for e in self.evaluations if e["run_id"] == run_id), None
        )
        self.selected_trace = None
        self.current_trace_data = None
        self._clear_processed_trace_data()

        if self.selected_run:
            self.load_traces()

    def load_traces(self):
        """Load traces for selected run."""
        if not self.selected_env or not self.selected_run:
            return

        self.loading = True
        self.error = None
        try:
            df = list_traces(self.selected_run["run_id"])
            self.traces = df.to_dict("records")
        except Exception as e:
            self.error = f"Failed to load traces: {str(e)}"
            self.traces = []
        finally:
            self.loading = False

    def select_trace(self, trace_id: str):
        """Select and load a specific trace."""
        self.selected_trace = next(
            (
                t
                for t in self.traces
                if str(t.get("trace_identifier", t.get("trace_id", "")))
                == str(trace_id)
            ),
            None,
        )

        if self.selected_trace:
            self.load_trace_data()

    def load_trace_data(self):
        """Load full trace data."""
        if not self.selected_env or not self.selected_run or not self.selected_trace:
            return

        self.loading = True
        self.error = None
        try:
            # Try database first
            trace_id = str(
                self.selected_trace.get(
                    "trace_identifier", self.selected_trace.get("trace_id", "")
                )
            )
            trace_data = get_trace(trace_id)
            if not trace_data:
                # Fallback to filesystem
                trace_data = get_trace_from_file(
                    self.selected_env, self.selected_run["run_id"], trace_id
                )

            if trace_data:
                self.current_trace_data = trace_data
                self._process_trace_data()
            else:
                self.error = "Trace data not found"
        except Exception as e:
            self.error = f"Failed to load trace: {str(e)}"
        finally:
            self.loading = False

    def _clear_processed_trace_data(self):
        """Clear processed trace data."""
        self.trace_model_name = ""
        self.trace_difficulty = ""
        self.trace_total_reward = 0.0
        self.trace_num_turns = 0
        self.trace_turns = []

    def _process_trace_data(self):
        """Process raw trace data into typed state variables."""
        if not self.current_trace_data or not self.selected_env:
            return

        # Use the new framework to process trace data
        processed = process_trace_for_environment(
            self.selected_env, self.current_trace_data
        )

        self.trace_model_name = processed["model_name"]
        self.trace_difficulty = processed["difficulty"]
        self.trace_total_reward = processed["total_reward"]
        self.trace_num_turns = processed["num_turns"]
        self.trace_turns = processed["turns"]

    def toggle_raw_data(self):
        """Toggle raw data visibility."""
        self.show_raw_data = not self.show_raw_data

    @rx.var
    def trace_data_json(self) -> str:
        """Return trace data as formatted JSON string."""
        if self.current_trace_data:
            return json.dumps(self.current_trace_data, indent=2)
        return "{}"

    @rx.var
    def is_crafter_env(self) -> bool:
        """Check if current environment is crafter."""
        return self.selected_env == "crafter"

    @rx.var
    def is_nethack_env(self) -> bool:
        """Check if current environment is nethack."""
        return self.selected_env == "nethack"

    @rx.var
    def is_minigrid_env(self) -> bool:
        """Check if current environment is minigrid."""
        return self.selected_env == "minigrid"


def evaluation_card(run: Dict) -> rx.Component:
    """Create an evaluation card component."""
    return rx.card(
        rx.vstack(
            rx.text(run["run_id"], font_weight="bold"),
            rx.text(
                f"Models: {run.get('models_evaluated', 'N/A')}", font_size="0.9rem"
            ),
            rx.text(
                f"Trajectories: {run.get('num_trajectories', 0)}", font_size="0.9rem"
            ),
            spacing="1",
        ),
        on_click=lambda: State.select_run(run["run_id"]),
        cursor="pointer",
        _hover={"bg": "gray.100"},
        padding="0.5rem",
        margin_bottom="0.5rem",
    )


def trace_card(trace: Dict) -> rx.Component:
    """Create a trace card component."""
    trace_id = trace.get("trace_identifier", trace.get("trace_id", ""))
    return rx.card(
        rx.hstack(
            rx.text(trace_id, font_weight="bold"),
            rx.text(trace.get("model_name", "N/A")),
            rx.text(f"Reward: {trace.get('total_reward', 0):.3f}"),
            rx.text(f"Steps: {trace.get('num_steps', 0)}"),
            spacing="4",
        ),
        on_click=lambda: State.select_trace(str(trace_id)),
        cursor="pointer",
        _hover={"bg": "gray.100"},
        padding="0.5rem",
        margin_bottom="0.5rem",
    )


def sidebar() -> rx.Component:
    """Left sidebar with environment and run selection."""
    return rx.vstack(
        rx.heading("Environments", size="4"),
        rx.select(
            State.environments,
            placeholder="Select environment",
            value=State.selected_env,
            on_change=State.select_environment,
            width="100%",
        ),
        rx.cond(
            State.selected_env,
            rx.vstack(
                rx.heading("Evaluation Runs", size="4", margin_top="1rem"),
                rx.cond(
                    State.evaluations,
                    rx.vstack(
                        rx.foreach(
                            State.evaluations,
                            evaluation_card,
                        ),
                        width="100%",
                    ),
                    rx.text("No evaluation runs found", color="gray.500"),
                ),
                width="100%",
            ),
        ),
        width="300px",
        padding="1rem",
        bg="gray.50",
        height="100vh",
        overflow_y="auto",
    )


def trace_list() -> rx.Component:
    """List of traces for selected run."""
    return rx.cond(
        State.selected_run,
        rx.vstack(
            rx.heading(f"Traces for {State.selected_run['run_id']}", size="3"),
            rx.cond(
                State.traces,
                rx.vstack(
                    rx.foreach(
                        State.traces,
                        trace_card,
                    ),
                    width="100%",
                ),
                rx.text("No traces found", color="gray.500"),
            ),
            width="100%",
            padding="1rem",
        ),
        rx.text(
            "Select an evaluation run to view traces", color="gray.500", padding="1rem"
        ),
    )


def render_crafter_turn(turn_data: Dict) -> rx.Component:
    """Render a single Crafter turn."""
    return rx.vstack(
        # Turn header
        rx.hstack(
            rx.text(
                f"Turn {turn_data['turn_number']}",
                font_weight="bold",
                font_size="1.1rem",
            ),
            width="100%",
        ),
        # Note: Achievements would be shown here (simplified for Reflex compatibility)
        # Actions (simplified)
        rx.text("Actions: Available", font_size="0.9rem", color="blue.600"),
        # Stats
        rx.cond(
            turn_data["stats"],
            rx.hstack(
                rx.text("Stats:", font_weight="bold", font_size="0.9rem"),
                rx.badge(
                    f"❤️ Health: {turn_data['stats']['health']}", color_scheme="red"
                ),
                rx.badge(
                    f"🍖 Food: {turn_data['stats']['food']}", color_scheme="orange"
                ),
                rx.badge(
                    f"💧 Drink: {turn_data['stats']['drink']}", color_scheme="blue"
                ),
                spacing="2",
            ),
        ),
        # Images (simplified)
        rx.text("Game images: Available", font_size="0.9rem", color="green.600"),
        spacing="3",
        padding="1rem",
        border="1px solid #eee",
        border_radius="8px",
        margin_bottom="1rem",
    )


def render_nethack_turn(turn_data: Dict) -> rx.Component:
    """Render a single NetHack turn."""
    return rx.vstack(
        # Turn header
        rx.text(
            f"Turn {turn_data['turn_number']}", font_weight="bold", font_size="1.1rem"
        ),
        # Actions
        rx.cond(
            turn_data["actions"],
            rx.vstack(
                rx.text("Actions:", font_weight="bold", font_size="0.9rem"),
                rx.foreach(
                    turn_data["actions"],
                    lambda action: rx.badge(
                        action["display_name"], color_scheme="purple"
                    ),
                ),
                spacing="2",
            ),
        ),
        # Stats
        rx.cond(
            turn_data["stats"],
            rx.hstack(
                rx.text("Stats:", font_weight="bold", font_size="0.9rem"),
                rx.badge(
                    f"HP: {turn_data['stats']['hp']}/{turn_data['stats']['max_hp']}",
                    color_scheme="red",
                ),
                rx.badge(f"Level: {turn_data['stats']['level']}", color_scheme="blue"),
                rx.badge(f"Gold: {turn_data['stats']['gold']}", color_scheme="yellow"),
                rx.badge(f"AC: {turn_data['stats']['ac']}", color_scheme="green"),
                spacing="2",
            ),
        ),
        # Messages
        rx.cond(
            turn_data["metadata"]["messages"],
            rx.vstack(
                rx.text("Messages:", font_weight="bold", font_size="0.9rem"),
                rx.box(
                    rx.foreach(
                        turn_data["metadata"]["messages"],
                        lambda msg: rx.text(
                            msg, font_family="monospace", font_size="0.85rem"
                        ),
                    ),
                    padding="0.5rem",
                    bg="gray.100",
                    border_radius="4px",
                ),
                spacing="2",
            ),
        ),
        spacing="3",
        padding="1rem",
        border="1px solid #eee",
        border_radius="8px",
        margin_bottom="1rem",
    )


def render_nethack_viewer() -> rx.Component:
    """Render NetHack-specific trace viewer."""
    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading(f"NetHack Trace - {State.trace_model_name}", size="4"),
            justify_content="space-between",
            width="100%",
        ),
        # Summary stats
        rx.hstack(
            rx.vstack(
                rx.text("Total Score", font_size="0.8rem", color="gray.600"),
                rx.text(
                    f"{State.trace_total_reward:.0f}",
                    font_size="1.5rem",
                    font_weight="bold",
                ),
                align_items="center",
                bg="gray.50",
                padding="1rem",
                border_radius="8px",
            ),
            rx.vstack(
                rx.text("Total Turns", font_size="0.8rem", color="gray.600"),
                rx.text(State.trace_num_turns, font_size="1.5rem", font_weight="bold"),
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
            rx.foreach(
                State.trace_turns,
                render_nethack_turn,
            ),
            width="100%",
        ),
        width="100%",
        spacing="4",
    )


def render_crafter_viewer() -> rx.Component:
    """Render Crafter-specific trace viewer."""
    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading(f"Crafter Trace - {State.trace_model_name}", size="4"),
            rx.badge(f"Difficulty: {State.trace_difficulty}", color_scheme="blue"),
            justify_content="space-between",
            width="100%",
        ),
        # Summary stats
        rx.hstack(
            rx.vstack(
                rx.text("Total Reward", font_size="0.8rem", color="gray.600"),
                rx.text(
                    f"{State.trace_total_reward:.3f}",
                    font_size="1.5rem",
                    font_weight="bold",
                ),
                align_items="center",
                bg="gray.50",
                padding="1rem",
                border_radius="8px",
            ),
            rx.vstack(
                rx.text("Total Turns", font_size="0.8rem", color="gray.600"),
                rx.text(State.trace_num_turns, font_size="1.5rem", font_weight="bold"),
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
            rx.foreach(
                State.trace_turns,
                render_crafter_turn,
            ),
            width="100%",
        ),
        width="100%",
        spacing="4",
    )


def render_minigrid_turn(turn_data: Dict) -> rx.Component:
    """Render a single MiniGrid turn."""
    return rx.vstack(
        # Turn header
        rx.text(
            f"Step {turn_data['turn_number']}", font_weight="bold", font_size="1.1rem"
        ),
        # Mission
        rx.cond(
            turn_data["metadata"]["mission"],
            rx.vstack(
                rx.text("Mission:", font_weight="bold", font_size="0.9rem"),
                rx.text(
                    turn_data["metadata"]["mission"],
                    font_style="italic",
                    color="gray.700",
                ),
                spacing="1",
            ),
        ),
        # Actions
        rx.cond(
            turn_data["actions"],
            rx.vstack(
                rx.text("Action:", font_weight="bold", font_size="0.9rem"),
                rx.foreach(
                    turn_data["actions"],
                    lambda action: rx.badge(
                        action["display_name"], color_scheme="blue"
                    ),
                ),
                spacing="2",
            ),
        ),
        # Rewards
        rx.cond(
            turn_data["metadata"]["rewards"],
            rx.hstack(
                rx.text("Rewards:", font_weight="bold", font_size="0.9rem"),
                rx.foreach(
                    turn_data["metadata"]["rewards"],
                    lambda r: rx.cond(
                        r != 0,
                        rx.badge(f"{r:+.2f}", color_scheme="green" if r > 0 else "red"),
                        rx.text(""),
                    ),
                ),
                spacing="2",
            ),
        ),
        # Images (simplified)
        rx.text("Grid visualization: Available", font_size="0.9rem", color="green.600"),
        spacing="3",
        padding="1rem",
        border="1px solid #eee",
        border_radius="8px",
        margin_bottom="1rem",
    )


def render_minigrid_viewer() -> rx.Component:
    """Render MiniGrid-specific trace viewer."""
    return rx.vstack(
        # Header
        rx.hstack(
            rx.heading(f"MiniGrid Trace - {State.trace_model_name}", size="4"),
            justify_content="space-between",
            width="100%",
        ),
        # Summary stats
        rx.hstack(
            rx.vstack(
                rx.text("Total Reward", font_size="0.8rem", color="gray.600"),
                rx.text(
                    f"{State.trace_total_reward:.2f}",
                    font_size="1.5rem",
                    font_weight="bold",
                ),
                align_items="center",
                bg="gray.50",
                padding="1rem",
                border_radius="8px",
            ),
            rx.vstack(
                rx.text("Total Steps", font_size="0.8rem", color="gray.600"),
                rx.text(State.trace_num_turns, font_size="1.5rem", font_weight="bold"),
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
            rx.heading("Step Details", size="4"),
            rx.foreach(
                State.trace_turns,
                render_minigrid_turn,
            ),
            width="100%",
        ),
        width="100%",
        spacing="4",
    )


def render_generic_viewer() -> rx.Component:
    """Render generic trace viewer for environments without specific viewers."""
    return rx.vstack(
        rx.heading("Summary", size="4"),
        rx.box(
            rx.vstack(
                rx.text(
                    f"Trace ID: {State.selected_trace.get('trace_identifier', State.selected_trace.get('trace_id', 'N/A'))}"
                ),
                rx.text(f"Model: {State.selected_trace.get('model_name', 'N/A')}"),
                rx.text(
                    f"Total Reward: {State.selected_trace.get('total_reward', 0):.3f}"
                ),
                rx.text(f"Steps: {State.selected_trace.get('num_steps', 0)}"),
                spacing="2",
            ),
            padding="1rem",
            bg="gray.50",
        ),
        spacing="4",
    )


def trace_viewer() -> rx.Component:
    """Display selected trace data with environment-specific rendering."""
    return rx.cond(
        State.current_trace_data,
        rx.vstack(
            rx.heading("Trace Details", size="3"),
            # Environment-specific viewer
            rx.cond(
                State.is_crafter_env,
                render_crafter_viewer(),
                rx.cond(
                    State.is_nethack_env,
                    render_nethack_viewer(),
                    rx.cond(
                        State.is_minigrid_env,
                        render_minigrid_viewer(),
                        render_generic_viewer(),
                    ),
                ),
            ),
            # Collapsible raw data section
            rx.vstack(
                rx.button(
                    rx.cond(
                        State.show_raw_data,
                        "Hide Raw Trace Data",
                        "Show Raw Trace Data",
                    ),
                    on_click=State.toggle_raw_data,
                    margin_top="2rem",
                ),
                rx.cond(
                    State.show_raw_data,
                    rx.box(
                        rx.code(
                            State.trace_data_json,
                            language="json",
                            font_size="0.9rem",
                        ),
                        height="400px",
                        overflow_y="auto",
                        bg="gray.50",
                        padding="1rem",
                        margin_top="0.5rem",
                    ),
                ),
            ),
            width="100%",
            padding="1rem",
            spacing="4",
        ),
        rx.center(
            rx.text("Select a trace to view details", color="gray.500"),
            height="400px",
        ),
    )


def main_content() -> rx.Component:
    """Main content area."""
    return rx.vstack(
        rx.cond(
            State.loading,
            rx.center(
                rx.spinner(size="3"),
                padding="2rem",
            ),
            rx.vstack(
                rx.cond(
                    State.error,
                    rx.callout(
                        State.error,
                        icon="triangle_alert",
                        color_scheme="red",
                    ),
                ),
                trace_list(),
                rx.divider(),
                trace_viewer(),
                spacing="4",
            ),
        ),
        width="100%",
        height="100vh",
        overflow_y="auto",
    )


def index() -> rx.Component:
    """Main page layout."""
    return rx.hstack(
        sidebar(),
        main_content(),
        spacing="0",
        width="100%",
        height="100vh",
        on_mount=State.load_environments,
    )


# Create the app
app = rx.App()

# Add pages
app.add_page(index, title="Synth Trace Viewer")
