"""
Base class for environment-specific viewers with proper error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseEnvironmentViewer(ABC):
    """
    Base class that each environment must implement.
    NO ASSUMPTIONS - each environment must explicitly define its behavior.
    """
    
    @abstractmethod
    def get_environment_name(self) -> str:
        """Return the name of this environment (e.g., 'crafter', 'nethack')."""
        pass
    
    @abstractmethod
    def validate_trace_structure(self, trace_data: Dict[str, Any]) -> None:
        """
        Validate that a trace has the expected structure for this environment.
        Should raise AssertionError with descriptive message if invalid.
        """
        pass
    
    @abstractmethod
    def get_trace_summary(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract summary information from a trace.
        Return dict with at least: {
            'num_turns': int,
            'model_name': str,
            'total_reward': float,
            'custom_stats': {...}  # Environment-specific stats
        }
        """
        pass
    
    @abstractmethod
    def render_turn_html(self, trace_data: Dict[str, Any], turn_index: int) -> str:
        """
        Render HTML for a specific turn.
        This is where environment-specific visualizations go (images, grids, etc).
        """
        pass
    
    @abstractmethod
    def get_custom_css(self) -> str:
        """Return any custom CSS needed for this environment's visualizations."""
        pass
    
    @abstractmethod
    def get_custom_javascript(self) -> str:
        """Return any custom JavaScript needed for this environment's interactions."""
        pass
    
    def extract_turn_metadata(self, trace_data: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
        """
        Extract metadata for a turn. Can be overridden for custom behavior.
        Default implementation that can be overridden.
        """
        try:
            partition = trace_data['trace']['partition'][turn_index]
            event = partition['events'][0] if partition.get('events') else {}
            
            return {
                'event_type': event.get('event_type', 'unknown'),
                'has_agent_step': 'agent_compute_step' in event,
                'num_env_steps': len(event.get('environment_compute_steps', [])),
                'raw_event': event  # For debugging
            }
        except Exception as e:
            return {'error': str(e)}


class EnvironmentViewerRegistry:
    """Registry for environment viewers with proper error handling."""
    
    def __init__(self):
        self._viewers: Dict[str, BaseEnvironmentViewer] = {}
    
    def register(self, viewer: BaseEnvironmentViewer) -> None:
        """Register an environment viewer."""
        name = viewer.get_environment_name()
        assert name not in self._viewers, f"Viewer for '{name}' already registered"
        self._viewers[name] = viewer
        print(f"✅ Registered viewer for environment: {name}")
    
    def get(self, env_name: str) -> Optional[BaseEnvironmentViewer]:
        """Get viewer for an environment."""
        return self._viewers.get(env_name)
    
    def list_environments(self) -> List[str]:
        """List all registered environments."""
        return list(self._viewers.keys())
    
    def detect_environment(self, eval_dir: Path) -> Optional[str]:
        """
        Try to detect which environment an evaluation is for.
        Returns environment name or None.
        """
        # Check parent directory name
        parent_name = eval_dir.parent.name.lower()
        if parent_name in self._viewers:
            return parent_name
        
        # Check for environment-specific files
        # Each viewer could implement a detect() method if needed
        
        return None


# Global registry instance
viewer_registry = EnvironmentViewerRegistry()