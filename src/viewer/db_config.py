"""
Database configuration module with absolute path handling and validation.
"""

import os
import pathlib
from typing import Optional


class DatabaseConfig:
    """Database configuration with absolute path handling and validation."""

    def __init__(self):
        self._db_path: Optional[pathlib.Path] = None
        self._initialize_db_path()

    def _initialize_db_path(self):
        """Initialize database path with proper validation."""
        # First check environment variable
        env_db_path = os.getenv("TRACE_DB")
        if env_db_path:
            self._db_path = pathlib.Path(env_db_path).resolve()
            return

        # Get the absolute path to the project root
        # This file is at src/viewer/db_config.py
        current_file = pathlib.Path(__file__).resolve()
        viewer_dir = current_file.parent  # src/viewer
        src_dir = viewer_dir.parent  # src
        project_root = src_dir.parent  # project root

        # Check for database in multiple locations
        possible_paths = [
            project_root / "synth_eval.duckdb",  # Root directory
            viewer_dir / "synth_eval.duckdb",  # Viewer directory
            project_root / "synth_eval_viewer.duckdb",  # Alternative name
        ]

        for path in possible_paths:
            if path.exists():
                self._db_path = path
                return

        # Default to root directory path
        self._db_path = project_root / "synth_eval.duckdb"

    @property
    def db_path(self) -> str:
        """Get the database path as a string."""
        assert self._db_path is not None, "Database path not initialized"
        return str(self._db_path)

    @property
    def db_path_absolute(self) -> pathlib.Path:
        """Get the database path as an absolute Path object."""
        assert self._db_path is not None, "Database path not initialized"
        return self._db_path

    def validate_db_exists(self) -> bool:
        """Validate that the database file exists."""
        return self._db_path.exists() if self._db_path else False

    def get_db_info(self) -> dict:
        """Get information about the database configuration."""
        return {
            "path": str(self._db_path) if self._db_path else None,
            "exists": self.validate_db_exists(),
            "absolute_path": str(self._db_path.resolve()) if self._db_path else None,
            "parent_dir": str(self._db_path.parent) if self._db_path else None,
            "env_var": os.getenv("TRACE_DB"),
        }


# Global instance
db_config = DatabaseConfig()
