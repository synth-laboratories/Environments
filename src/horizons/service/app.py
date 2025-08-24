import sys
import os

# Ensure local 'src' directory is on PYTHONPATH for dev installs
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

# Register built-in environments at import time (best-effort, guard missing ones)
try:
    import horizons.examples.sokoban.environment as sok
    register_environment("Sokoban", sok.SokobanEnvironment)
except Exception as e:
    logger.warning(f"Sokoban not available: {e}")

# Optional: register PyO3-backed Sokoban variant
try:
    import horizons.examples.sokoban_pyo3.environment as sok_pyo3
    register_environment("Sokoban_PyO3", sok_pyo3.SokobanPyO3Environment)
except Exception as e:
    logger.info(f"Sokoban_PyO3 not available: {e}")

# The following are optional; guard with try/except to avoid import-time failures
for name, modpath, clsname in [
    ("CrafterClassic", "horizons.examples.crafter_classic.environment", "CrafterClassicEnvironment"),
    ("MiniGrid", "horizons.examples.minigrid.environment", "MiniGridEnvironment"),
    ("TicTacToe", "horizons.examples.tictactoe.environment", "TicTacToeEnvironment"),
    ("Verilog", "horizons.examples.verilog.environment", "VerilogEnvironment"),
    ("HendryksMath", "horizons.examples.math.environment", "HendryksMathEnv"),
    ("NetHack", "horizons.examples.nethack.environment", "NetHackEnvironment"),
    ("Enron", "horizons.examples.enron.environment", "EnronEnvironment"),
]:
    try:
        mod = __import__(modpath, fromlist=[clsname])
        register_environment(name, getattr(mod, clsname))
    except Exception as e:
        logger.info(f"{name} not available: {e}")

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
