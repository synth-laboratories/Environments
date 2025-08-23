"""
Horizons AI - Advanced Reinforcement Learning Environments

A comprehensive collection of environments for reinforcement learning research and development.
"""

__version__ = "0.1.0"
__author__ = "Synth AI"
__email__ = "josh@usesynth.ai"

# Import tracing functionality from synth-ai (with fallbacks for different versions)
try:
    from synth_ai import trace, tracer, TracingConfig
except ImportError:
    # Fallback for older synth-ai versions
    try:
        from synth_ai.tracing_v3 import *
        from synth_ai import tracer
        trace = None  # Not available in older versions
        TracingConfig = None  # Not available in older versions
    except ImportError:
        # Minimal fallback - tracing not available
        trace = None
        tracer = None
        TracingConfig = None

# Import local environments
from .environments import *

# Make key modules available
__all__ = [
    # Tracing functionality (imported from synth-ai)
    "trace",
    "tracer",
    "TracingConfig",

    # Local environments
    "environments",

    # Local modules
    "environment",
    "service",
    "stateful",
    "tasks",
    "examples",
]
