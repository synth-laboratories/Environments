import sys
import os

# Ensure local 'src' directory is on PYTHONPATH for dev installs
# Add <repo>/src to sys.path (two levels up)
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
import logging

from fastapi import FastAPI
from .registry import list_supported_env_types, register_environment
from .core_routes import api_router
from .external_registry import (
    ExternalRegistryConfig,
    load_external_environments,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Register built-in environments at import time.
We keep registrations best-effort so the service can start without optional deps."""

# Sokoban (pure)
try:
    import horizons.examples.sokoban.environment as sok
    register_environment("Sokoban", sok.SokobanEnvironment)
except Exception as e:
    logger.warning(f"Sokoban not available: {e}")

# Sokoban (PyO3-backed)
try:
    import horizons.examples.sokoban_pyo3.environment as sok_pyo3
    register_environment("Sokoban_PyO3", sok_pyo3.SokobanPyO3Environment)
    logger.info("Registered PyO3-backed Sokoban environment as 'Sokoban_PyO3'")
except Exception as e:
    logger.info(f"Sokoban_PyO3 not available: {e}")

try:
    import horizons.examples.crafter_classic.environment as cc
    register_environment("CrafterClassic", cc.CrafterClassicEnvironment)
except Exception as e:
    logger.info(f"CrafterClassic not available: {e}")

try:
    import horizons.examples.minigrid.environment as mg
    register_environment("MiniGrid", mg.MiniGridEnvironment)
except Exception as e:
    logger.info(f"MiniGrid not available: {e}")

# MiniGrid (PyO3-backed)
try:
    import horizons.examples.minigrid_pyo3.environment as mg_pyo3
    register_environment("MiniGrid_PyO3", mg_pyo3.MiniGridPyO3Environment)
    logger.info("Registered PyO3-backed MiniGrid environment as 'MiniGrid_PyO3'")
except Exception as e:
    logger.info(f"MiniGrid_PyO3 not available: {e}")

# TicTacToe (register if present in repo)
for name, modpath, clsname in [
    ("TicTacToe", "horizons.examples.tictactoe.environment", "TicTacToeEnvironment"),
    ("TicTacToe_PyO3", "horizons.examples.tictactoe_pyo3.environment", "TicTacToePyO3Environment"),
]:
    try:
        mod = __import__(modpath, fromlist=[clsname])
        register_environment(name, getattr(mod, clsname))
        logger.info(f"Registered {name}")
    except Exception as e:
        logger.info(f"{name} not available: {e}")

# Note: Enron environment is not included in this version
# as it was removed during refactoring

app = FastAPI(title="Environment Service")


@app.on_event("startup")
async def startup_event():
    """Load external environments on startup."""
    # Support configuration-based loading for external environments
    # You can set EXTERNAL_ENVIRONMENTS env var with JSON config
    external_config = os.getenv("EXTERNAL_ENVIRONMENTS")
    if external_config:
        try:
            import json

            config_data = json.loads(external_config)
            config = ExternalRegistryConfig(
                external_environments=config_data.get("external_environments", [])
            )
            load_external_environments(config)
        except Exception as e:
            logger.error(f"Failed to load external environment config: {e}")

    # Log all registered environments
    logger.info(f"Registered environments: {list_supported_env_types()}")


# Mount the main API router
app.include_router(api_router, tags=["environments"])
