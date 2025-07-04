import reflex as rx
import json
import asyncio
from typing import Dict, List, Optional, Any
import sys
import os
from pathlib import Path

# Import database functions from local module
from .database import (
    list_evaluations, list_traces, get_trace, 
    get_trace_from_file, get_environments
)

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
    
    # WebSocket state
    connected: bool = False
    recent_updates: List[Dict] = []
    
    async def load_environments(self):
        """Load available environments."""
        self.loading = True
        self.error = None
        try:
            self.environments = get_environments()
        except Exception as e:
            self.error = f"Failed to load environments: {str(e)}"
        finally:
            self.loading = False
    
    async def select_environment(self, env_name: str):
        """Select an environment and load its evaluations."""
        self.selected_env = env_name
        self.selected_run = None
        self.selected_trace = None
        self.current_trace_data = None
        await self.load_evaluations()
    
    async def load_evaluations(self):
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
        finally:
            self.loading = False
    
    async def select_run(self, run_id: str):
        """Select a run and load its traces."""
        self.selected_run = next(
            (e for e in self.evaluations if e["run_id"] == run_id), 
            None
        )
        self.selected_trace = None
        self.current_trace_data = None
        
        if self.selected_run:
            await self.load_traces()
    
    async def load_traces(self):
        """Load traces for selected run."""
        if not self.selected_env or not self.selected_run:
            return
            
        self.loading = True
        self.error = None
        try:
            df = list_traces(self.selected_run['run_id'])
            self.traces = df.to_dict("records")
        except Exception as e:
            self.error = f"Failed to load traces: {str(e)}"
        finally:
            self.loading = False
    
    async def select_trace(self, trace_id: str):
        """Select and load a specific trace."""
        self.selected_trace = next(
            (t for t in self.traces if t["trace_id"] == trace_id),
            None
        )
        
        if self.selected_trace:
            await self.load_trace_data()
    
    async def load_trace_data(self):
        """Load full trace data."""
        if not self.selected_env or not self.selected_run or not self.selected_trace:
            return
            
        self.loading = True
        self.error = None
        try:
            # Try database first
            trace_data = get_trace(self.selected_trace['trace_id'])
            if not trace_data:
                # Fallback to filesystem
                trace_data = get_trace_from_file(
                    self.selected_env, 
                    self.selected_run['run_id'], 
                    self.selected_trace['trace_id']
                )
            
            if trace_data:
                self.current_trace_data = trace_data
            else:
                self.error = "Trace data not found"
        except Exception as e:
            self.error = f"Failed to load trace: {str(e)}"
        finally:
            self.loading = False


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
                    State.evaluations.length() > 0,
                    rx.vstack(
                        rx.foreach(
                            State.evaluations,
                            lambda run: rx.card(
                                rx.vstack(
                                    rx.text(run["run_id"], font_weight="bold"),
                                    rx.text(f"Models: {run['models_evaluated']}", font_size="0.9rem"),
                                    rx.text(f"Success: {run['success_rate']:.1%}", font_size="0.9rem"),
                                    rx.text(f"Trajectories: {run['num_trajectories']}", font_size="0.9rem"),
                                    spacing="1",
                                ),
                                on_click=lambda: State.select_run(run["run_id"]),
                                cursor="pointer",
                                _hover={"bg": "gray.100"},
                                padding="0.5rem",
                                margin_bottom="0.5rem",
                            )
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
                State.traces.length() > 0,
                rx.vstack(
                    # Header
                    rx.hstack(
                        rx.text("Trace ID", font_weight="bold", width="200px"),
                        rx.text("Model", font_weight="bold", width="150px"),
                        rx.text("Reward", font_weight="bold", width="100px"),
                        rx.text("Steps", font_weight="bold", width="100px"),
                        spacing="4",
                        padding="0.5rem",
                        bg="gray.200",
                    ),
                    # Data rows
                    rx.foreach(
                        State.traces,
                        lambda trace: rx.card(
                            rx.hstack(
                                rx.text(trace["trace_id"], font_weight="bold", width="200px"),
                                rx.text(trace["model_name"], width="150px"),
                                rx.text(f"{trace['total_reward']:.3f}", width="100px"),
                                rx.text(f"{trace['num_steps']}", width="100px"),
                                spacing="4",
                            ),
                            on_click=lambda: State.select_trace(trace["trace_id"]),
                            cursor="pointer",
                            _hover={"bg": "gray.100"},
                            padding="0.5rem",
                            margin_bottom="0.5rem",
                        )
                    ),
                    width="100%",
                ),
                rx.text("No traces found", color="gray.500"),
            ),
            width="100%",
            padding="1rem",
        ),
        rx.text("Select an evaluation run to view traces", color="gray.500", padding="1rem"),
    )


def trace_viewer() -> rx.Component:
    """Dynamic trace viewer that switches based on environment."""
    return rx.cond(
        State.current_trace_data,
        default_viewer(),
        rx.center(
            rx.text("Select a trace to view details", color="gray.500"),
            height="400px",
        ),
    )


def default_viewer() -> rx.Component:
    """Default JSON viewer for unknown environments."""
    return rx.vstack(
        rx.heading(f"Trace: {State.selected_trace['trace_id']}", size="3"),
        rx.vstack(
            rx.heading("Trace Summary", size="4"),
            rx.vstack(
                rx.text(f"Model: {State.selected_trace['model_name']}"),
                rx.text(f"Total Reward: {State.selected_trace['total_reward']:.3f}"),
                rx.text(f"Steps: {State.selected_trace['num_steps']}"),
                rx.text(f"Termination: {State.selected_trace.get('terminated_reason', 'N/A')}"),
                spacing="2",
                padding="1rem",
                bg="gray.50",
            ),
            rx.heading("Raw JSON", size="4", margin_top="1rem"),
            rx.code_block(
                State.current_trace_data.to(str),
                language="json",
                show_line_numbers=True,
                max_height="400px",
                overflow_y="auto",
            ),
            spacing="4",
        ),
        width="100%",
        padding="1rem",
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
app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        radius="medium",
        accent_color="blue",
    )
)

# Add pages
app.add_page(index, title="Synth Trace Viewer") 