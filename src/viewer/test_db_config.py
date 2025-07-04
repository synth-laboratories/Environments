"""
Unit tests for database configuration and path handling.
"""
import pytest
import os
import pathlib
import tempfile
import sys
from unittest.mock import patch, MagicMock

# Add viewer directory to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from db_config import DatabaseConfig


class TestDatabaseConfig:
    """Test database configuration with absolute paths."""
    
    def test_db_path_is_absolute(self):
        """Test that database path is always absolute."""
        config = DatabaseConfig()
        db_path = config.db_path
        
        # Assert path is absolute
        assert os.path.isabs(db_path), f"Database path must be absolute, got: {db_path}"
        
        # Assert Path object is also absolute
        path_obj = config.db_path_absolute
        assert path_obj.is_absolute(), f"Path object must be absolute, got: {path_obj}"
    
    def test_env_variable_override(self):
        """Test that TRACE_DB environment variable works."""
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Set environment variable
            with patch.dict(os.environ, {"TRACE_DB": tmp_path}):
                config = DatabaseConfig()
                
                # Should use the env variable path (resolve to handle symlinks)
                assert pathlib.Path(config.db_path).resolve() == pathlib.Path(tmp_path).resolve()
                assert config.validate_db_exists()
                
                # Should be absolute
                assert os.path.isabs(config.db_path)
        finally:
            os.unlink(tmp_path)
    
    def test_relative_env_path_becomes_absolute(self):
        """Test that relative paths in TRACE_DB become absolute."""
        with patch.dict(os.environ, {"TRACE_DB": "./relative/path.duckdb"}):
            config = DatabaseConfig()
            
            # Path should be absolute
            assert os.path.isabs(config.db_path)
            assert config.db_path_absolute.is_absolute()
    
    def test_project_root_detection(self):
        """Test correct project root detection."""
        config = DatabaseConfig()
        
        # Clear env variable to test default behavior
        with patch.dict(os.environ, {"TRACE_DB": ""}, clear=True):
            config._initialize_db_path()
            
            # Should find project root correctly
            # db_config.py is at src/viewer/db_config.py
            expected_root = current_dir.parent.parent  # Go up to project root
            actual_root = config.db_path_absolute.parent
            
            # The database should be in or near the project root
            assert expected_root in actual_root.parents or actual_root == expected_root
    
    def test_db_info_contains_all_fields(self):
        """Test that get_db_info returns all expected fields."""
        config = DatabaseConfig()
        info = config.get_db_info()
        
        # Check all required fields exist
        assert "path" in info
        assert "exists" in info
        assert "absolute_path" in info
        assert "parent_dir" in info
        assert "env_var" in info
        
        # Check types
        assert isinstance(info["exists"], bool)
        assert info["path"] is None or isinstance(info["path"], str)
        
        # If path exists, it should be absolute
        if info["path"]:
            assert os.path.isabs(info["absolute_path"])
    
    def test_validate_db_exists(self):
        """Test database existence validation."""
        config = DatabaseConfig()
        
        # Create a temp file to test existence
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch.dict(os.environ, {"TRACE_DB": tmp_path}):
                config = DatabaseConfig()
                assert config.validate_db_exists() is True
            
            # Remove file and test again
            os.unlink(tmp_path)
            assert config.validate_db_exists() is False
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_assertion_on_uninitialized_path(self):
        """Test that accessing path without initialization raises assertion."""
        config = DatabaseConfig()
        config._db_path = None  # Force uninitialized state
        
        with pytest.raises(AssertionError, match="Database path not initialized"):
            _ = config.db_path
        
        with pytest.raises(AssertionError, match="Database path not initialized"):
            _ = config.db_path_absolute


class TestDatabaseConnection:
    """Test database connection with path validation."""
    
    @patch('duckdb.connect')
    def test_connection_with_valid_db(self, mock_connect):
        """Test connection when database exists."""
        # Create a temporary database file
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch.dict(os.environ, {"TRACE_DB": tmp_path}):
                # Import here to use patched environment
                from db_config import db_config
                db_config._initialize_db_path()  # Reinitialize with new env
                
                # Import database module from backend
                from backend import database
                
                # Should connect successfully
                database.get_connection()
                mock_connect.assert_called_once()
        finally:
            os.unlink(tmp_path)
    
    def test_connection_with_missing_db(self):
        """Test connection error when database doesn't exist."""
        # Use a path that doesn't exist
        fake_path = "/tmp/nonexistent_test_db_12345.duckdb"
        
        with patch.dict(os.environ, {"TRACE_DB": fake_path}):
            # Import here to use patched environment
            from db_config import db_config
            db_config._initialize_db_path()  # Reinitialize with new env
            
            # Import database module from backend
            from backend import database
            
            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                database.get_connection()
            
            # Check error message contains useful information
            error_msg = str(exc_info.value)
            assert fake_path in error_msg
            assert "Database not found" in error_msg
            assert "TRACE_DB" in error_msg


class TestPathAssertions:
    """Test that all paths are properly validated with assertions."""
    
    def test_trace_path_assertions(self):
        """Test trace file path assertions."""
        # Create mock db_config
        mock_db_config = MagicMock()
        mock_db_config.db_path_absolute.parent = pathlib.Path("/test/project")
        
        with patch('backend.database.db_config', mock_db_config):
            from backend import database
            
            # This should work - creates absolute path
            result = database.get_trace_from_file("env1", "run1", "trace1")
            assert result is None  # File doesn't exist, but no assertion error
    
    def test_invalid_relative_paths_caught(self):
        """Test that relative paths trigger assertions."""
        # This test ensures our assertions catch any accidental relative paths
        config = DatabaseConfig()
        
        # The db_path property converts to string, which doesn't have is_absolute
        # So we need to test the actual assertion in the database module
        with patch('backend.database.db_config') as mock_config:
            mock_config.validate_db_exists.return_value = True
            mock_config.db_path = "relative/path.db"  # Relative path string
            
            from backend import database
            
            # Should raise assertion error
            with pytest.raises(AssertionError, match="must be absolute"):
                database.get_connection()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])