"""CLI subcommands for Horizons AI.

This package hosts modular commands and exposes a top-level Click group
named `cli` compatible with the pyproject entry point `horizons.cli:cli`.
"""

from __future__ import annotations

# Load environment variables from a local .env if present (repo root)
try:
    from dotenv import find_dotenv, load_dotenv

    # Source .env early so CLI subcommands inherit config; do not override shell
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional at runtime; proceed if unavailable
    pass


from .root import cli  # new canonical CLI entrypoint

# For now, we'll just have the serve command
# Additional subcommands can be added later
