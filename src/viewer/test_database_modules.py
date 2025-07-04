"""
Integration tests for frontend and backend database modules.
"""
import pytest
import os
import pathlib
import tempfile
import sys
import json
from unittest.mock import patch, MagicMock

# Add directories to path
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "frontend" / "traces_viewer"))
sys.path.insert(0, str(current_dir / "backend"))


class TestDatabaseModules:
    """Test both frontend and backend database modules."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Set environment variable
        os.environ["TRACE_DB"] = self.temp_db_path
    
    def teardown_method(self):
        """Cleanup test environment."""
        # Remove temp file
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
        
        # Clear environment variable
        if "TRACE_DB" in os.environ:
            del os.environ["TRACE_DB"]
    
    def test_frontend_backend_use_same_path(self):
        """Test that frontend and backend use the same database path."""
        # Import both modules
        from frontend.traces_viewer import database as frontend_db
        from backend import database as backend_db
        
        # Re-import db_config to pick up new env var
        from db_config import db_config
        db_config._initialize_db_path()
        
        # Both should use the same absolute path (resolve to handle symlinks)
        assert pathlib.Path(db_config.db_path).resolve() == pathlib.Path(self.temp_db_path).resolve()
        assert os.path.isabs(db_config.db_path)
    
    def test_absolute_path_assertions(self):
        """Test that path assertions work correctly."""
        from frontend.traces_viewer import database as frontend_db
        
        # Mock db_config to return relative path (should never happen)
        with patch('frontend.traces_viewer.database.db_config') as mock_config:
            mock_config.validate_db_exists.return_value = True
            mock_config.db_path = "relative/path.db"  # Relative path
            
            # Should raise assertion error
            with pytest.raises(AssertionError, match="must be absolute"):
                frontend_db.get_connection()
    
    def test_missing_database_error(self):
        """Test helpful error when database doesn't exist."""
        # Remove the temp database
        os.unlink(self.temp_db_path)
        
        # Re-import to pick up changes
        from frontend.traces_viewer import database as frontend_db
        from db_config import db_config
        db_config._initialize_db_path()
        
        # Should raise FileNotFoundError with helpful message
        with pytest.raises(FileNotFoundError) as exc_info:
            frontend_db.get_connection()
        
        error_msg = str(exc_info.value)
        assert "Database not found at:" in error_msg
        assert self.temp_db_path in error_msg
        assert "TRACE_DB" in error_msg
    
    def test_trace_file_paths_are_absolute(self):
        """Test that trace file paths are always absolute."""
        from frontend.traces_viewer import database as frontend_db
        from backend import database as backend_db
        
        # Test frontend
        result = frontend_db.get_trace_from_file("test_env", "run123", "trace456")
        # Even though file doesn't exist, path construction should work
        assert result is None
        
        # Test backend  
        result = backend_db.get_trace_from_file("test_env", "run123", "trace456")
        assert result is None
    
    def test_trace_file_loading(self):
        """Test loading trace files from filesystem."""
        from frontend.traces_viewer import database as frontend_db
        from db_config import db_config
        
        # Create test trace file structure
        project_root = pathlib.Path(self.temp_db_path).parent
        trace_dir = project_root / "evals" / "test_env" / "run123" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        
        trace_file = trace_dir / "trace456.json"
        trace_data = {"test": "data", "steps": [1, 2, 3]}
        
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f)
        
        try:
            # Mock db_config to use our test directory
            mock_db_config = MagicMock()
            mock_db_config.db_path_absolute.parent = project_root
            
            with patch('frontend.traces_viewer.database.db_config', mock_db_config):
                result = frontend_db.get_trace_from_file("test_env", "run123", "trace456")
                assert result == trace_data
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(project_root / "evals", ignore_errors=True)


class TestDatabaseConnectionPool:
    """Test database connection handling."""
    
    @patch('duckdb.connect')
    def test_connection_parameters(self, mock_connect):
        """Test that connections are opened with correct parameters."""
        # Setup
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch.dict(os.environ, {"TRACE_DB": tmp_path}):
                # Re-import to pick up env change
                from frontend.traces_viewer import database as frontend_db
                from db_config import db_config
                db_config._initialize_db_path()
                
                # Make connection
                frontend_db.get_connection()
                
                # Check connection was made with read_only=True
                # Use resolved path to handle symlinks
                resolved_path = str(pathlib.Path(tmp_path).resolve())
                mock_connect.assert_called_with(resolved_path, read_only=True)
        finally:
            os.unlink(tmp_path)


class TestErrorHandling:
    """Test error handling and user-friendly messages."""
    
    def test_clear_error_messages(self):
        """Test that errors provide clear guidance to users."""
        fake_path = "/nonexistent/path/db.duckdb"
        
        with patch.dict(os.environ, {"TRACE_DB": fake_path}):
            from frontend.traces_viewer import database as frontend_db
            from db_config import db_config
            db_config._initialize_db_path()
            
            with pytest.raises(FileNotFoundError) as exc_info:
                frontend_db.get_connection()
            
            error_msg = str(exc_info.value)
            
            # Check error message components
            assert "Database not found at:" in error_msg
            assert fake_path in error_msg
            assert "Environment variable TRACE_DB:" in error_msg
            assert "Please ensure the database file exists" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])