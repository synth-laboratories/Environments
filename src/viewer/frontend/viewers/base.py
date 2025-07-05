"""
Base framework for environment-specific trace viewers.

This module defines the interface and common patterns that environment-specific
viewers should follow to integrate with the main trace viewer.
"""

from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TraceImage:
    """Represents an image in a trace."""

    data_url: str
    caption: str
    step_number: int


@dataclass
class TraceAction:
    """Represents an action in a trace."""

    name: str
    index: int
    display_name: str


@dataclass
class TraceStats:
    """Represents stats for a trace turn."""

    data: Dict[str, Any]  # Flexible stats storage

    def get(self, key: str, default: Any = None) -> Any:
        """Get a stat value with default."""
        return self.data.get(key, default)


@dataclass
class TraceTurn:
    """Represents a single turn in a trace."""

    turn_number: int
    images: List[TraceImage]
    actions: List[TraceAction]
    stats: Optional[TraceStats]
    achievements: List[str]
    metadata: Dict[str, Any]  # Environment-specific metadata


class TraceProcessor(ABC):
    """Abstract base class for environment-specific trace processors."""

    @abstractmethod
    def process_turns(self, partitions: List[Dict]) -> List[Dict]:
        """
        Process raw partition data into serializable turn data.

        Args:
            partitions: Raw partition data from trace

        Returns:
            List of dictionaries representing processed turns
        """
        pass

    @abstractmethod
    def get_action_name(self, action_idx: int) -> str:
        """
        Map action index to human-readable name.

        Args:
            action_idx: Numeric action index

        Returns:
            Human-readable action name
        """
        pass

    def extract_metadata(self, trace_data: Dict) -> Dict[str, Any]:
        """
        Extract environment-specific metadata from trace.

        Args:
            trace_data: Full trace data

        Returns:
            Dictionary of metadata
        """
        trace = trace_data.get("trace", {})
        return trace.get("metadata", {})

    def extract_reward(self, trace_data: Dict) -> float:
        """
        Extract total reward from trace.

        Args:
            trace_data: Full trace data

        Returns:
            Total reward value
        """
        dataset = trace_data.get("dataset", {})
        reward_signals = dataset.get("reward_signals", [{}])
        if reward_signals:
            return reward_signals[0].get("reward", 0.0)
        return 0.0


class CrafterProcessor(TraceProcessor):
    """Crafter-specific trace processor."""

    def process_turns(self, partitions: List[Dict]) -> List[Dict]:
        """Process Crafter partition data into turn data."""
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
                    action_name = self.get_action_name(action_idx)

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
                    action_name = self.get_action_name(action_idx)
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

    def get_action_name(self, action_idx: int) -> str:
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


class NethackProcessor(TraceProcessor):
    """NetHack-specific trace processor."""

    def process_turns(self, partitions: List[Dict]) -> List[Dict]:
        """Process NetHack partition data into turn data."""
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
            glyphs = None

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

                # Extract glyphs
                if "glyphs" in outputs:
                    glyphs = outputs["glyphs"]

                # Extract action
                if "action" in outputs:
                    action = outputs["action"]
                    if isinstance(action, int):
                        action_name = self.get_action_name(action)
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
                        "messages": messages[-5:],  # Last 5 messages
                        "glyphs": glyphs,
                    },
                }
            )

        return turns

    def get_action_name(self, action_idx: int) -> str:
        """Map NetHack action index to name."""
        # Simplified - real implementation would have full action mapping
        return f"action_{action_idx}"


class GenericProcessor(TraceProcessor):
    """Generic trace processor for unknown environments."""

    def process_turns(self, partitions: List[Dict]) -> List[Dict]:
        """Process generic partition data."""
        turns = []
        for i, partition in enumerate(partitions[:5]):  # Fewer turns for generic
            turns.append(
                {
                    "turn_number": i + 1,
                    "images": [],
                    "actions": [],
                    "stats": None,
                    "achievements": [],
                    "metadata": {"raw_partition": partition},
                }
            )
        return turns

    def get_action_name(self, action_idx: int) -> str:
        """Generic action name."""
        return f"action_{action_idx}"


# Registry of processors
PROCESSORS = {
    "crafter": CrafterProcessor(),
    "nethack": NethackProcessor(),
}


def get_processor(env_name: str) -> TraceProcessor:
    """Get processor for environment or fallback to generic."""
    return PROCESSORS.get(env_name, GenericProcessor())


def process_trace_for_environment(env_name: str, trace_data: Dict) -> Dict[str, Any]:
    """
    Process trace data for a specific environment.

    Args:
        env_name: Environment name (e.g., 'crafter', 'nethack')
        trace_data: Raw trace data

    Returns:
        Processed trace data with typed fields
    """
    processor = get_processor(env_name)

    # Extract basic metadata
    metadata = processor.extract_metadata(trace_data)
    reward = processor.extract_reward(trace_data)

    # Process turns
    trace = trace_data.get("trace", {})
    partitions = trace.get("partition", [])
    turns = processor.process_turns(partitions)

    return {
        "model_name": metadata.get("model_name", "Unknown"),
        "difficulty": metadata.get("difficulty", "Unknown"),
        "total_reward": reward,
        "num_turns": len(partitions),
        "turns": turns,
    }
