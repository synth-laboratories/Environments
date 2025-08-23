"""
Integration tests for the Environment Service API.
Tests the new aligned API endpoints with various environments.
"""

import pytest
import json
from uuid import uuid4
from typing import Dict, Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from httpx import AsyncClient

from horizons.environments.service.app import app
from horizons.environments.service.core_routes import instances
from horizons.environments.examples.sokoban.units.astar_common import ENGINE_ASTAR


# Test fixtures
SIMPLE_SOKOBAN: Dict[str, Any] = {
    "dim_room": [4, 4],
    "room_fixed": [[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    "room_state": [[0, 0, 0, 0], [0, 1, 4, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
}


class TestServiceAPI:
    """Test the aligned Environment Service API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def async_client(self):
        """Create async test client."""
        return TestClient(app)

    def setup_method(self):
        """Clear instances before each test."""
        instances.clear()

    def teardown_method(self):
        """Clear instances after each test."""
        instances.clear()

    def test_health_endpoint(self, client):
        """Test the health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "supported_environments" in data
        assert "Sokoban" in data["supported_environments"]

    def test_initialize_endpoint(self, async_client):
        """Test the new initialize endpoint."""
        response = async_client.post(
            "/env/Sokoban/initialize",
            json={"initial_state": SIMPLE_SOKOBAN, "config": {}},
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "env_id" in data
        assert "observation" in data
        assert "done" in data
        assert "info" in data

        # Check observation content
        obs = data["observation"]
        assert "room_text" in obs
        assert "boxes_on_target" in obs
        assert obs["boxes_on_target"] == 0
        assert obs["max_steps"] == 10

        # Verify instance was stored
        assert data["env_id"] in instances

    
    async def test_step_endpoint(self, async_client):
        """Test the step endpoint."""
        # First initialize
        init_response = async_client.post(
            "/env/Sokoban/initialize", json={"initial_state": SIMPLE_SOKOBAN}
        )
        env_id = init_response.json()["env_id"]

        # Then step
        step_response = async_client.post(
            "/env/Sokoban/step",
            json={
                "env_id": env_id,
                "request_id": "test-request-123",
                "action": {
                    "tool_calls": [
                        {"tool": "interact", "args": {"action": 1}}  # DOWN
                    ]
                },
            },
        )

        assert step_response.status_code == 200
        data = step_response.json()

        # Check response structure
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

        # Check observation changed
        obs = data["observation"]
        assert "steps_taken" in obs
        assert obs["steps_taken"] == 1

    
    async def test_terminate_endpoint(self, async_client):
        """Test the terminate endpoint."""
        # Initialize first
        init_response = async_client.post(
            "/env/Sokoban/initialize", json={"initial_state": SIMPLE_SOKOBAN}
        )
        env_id = init_response.json()["env_id"]

        # Verify instance exists
        assert env_id in instances

        # Terminate
        term_response = async_client.post(
            "/env/Sokoban/terminate", json={"env_id": env_id}
        )

        assert term_response.status_code == 200
        data = term_response.json()

        assert "public" in data
        assert "private" in data
        assert data["private"]["instance_id"] == env_id

        # Verify instance was removed
        assert env_id not in instances

    
    async def test_invalid_environment_name(self, async_client):
        """Test handling of invalid environment names."""
        response = async_client.post(
            "/env/NonExistentEnv/initialize", json={"initial_state": {}}
        )

        assert response.status_code == 400
        assert "Unsupported environment type" in response.text

    
    async def test_invalid_env_id(self, async_client):
        """Test handling of invalid environment IDs."""
        fake_id = str(uuid4())

        # Try to step with non-existent ID
        response = async_client.post(
            "/env/Sokoban/step", json={"env_id": fake_id, "request_id": "test-invalid", "action": {"tool_calls": []}}
        )

        assert response.status_code == 404
        assert "not found" in response.text


    @pytest.mark.skip(reason="A* solver is async and requires complex conversion to sync")
    def test_complete_sokoban_episode(self, async_client):
        """Test a complete Sokoban episode using A* solver."""
        # Initialize
        init_response = async_client.post(
            "/env/Sokoban/initialize", json={"initial_state": SIMPLE_SOKOBAN}
        )
        env_id = init_response.json()["env_id"]

        # Get the environment instance to run A*
        env = instances[env_id]
        env.engine.package_sokoban_env.observation_mode = "raw"  # Disable rendering

        # Find solution with A*
        plan = ENGINE_ASTAR(env.engine, max_nodes=100)
        assert plan is not None
        assert len(plan) > 0

        # Execute plan via API
        for action in plan:
            step_response = async_client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": env_id,
                    "request_id": f"test-step-{action}",
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": action}}]
                    },
                },
            )
            assert step_response.status_code == 200

            data = step_response.json()
            obs = data["observation"]

            # Check if solved
            if obs.get("boxes_on_target") == SIMPLE_SOKOBAN["num_boxes"]:
                break

        # Verify puzzle was solved
        assert obs["boxes_on_target"] == SIMPLE_SOKOBAN["num_boxes"]

        # Clean up
        async_client.post("/env/Sokoban/terminate", json={"env_id": env_id})

    
    async def test_checkpoint_endpoint(self, async_client):
        """Test the checkpoint endpoint."""
        # Initialize
        init_response = async_client.post(
            "/env/Sokoban/initialize", json={"initial_state": SIMPLE_SOKOBAN}
        )
        env_id = init_response.json()["env_id"]

        # Get checkpoint
        response = async_client.get(f"/Sokoban/{env_id}/checkpoint")
        assert response.status_code == 200

        data = response.json()
        assert "snapshot" in data

        snapshot = data["snapshot"]
        assert "engine_snapshot_data" in snapshot

        engine_snapshot = snapshot["engine_snapshot_data"]
        assert "engine_snapshot" in engine_snapshot
        assert engine_snapshot["engine_snapshot"]["num_boxes"] == 1

    
    async def test_multiple_environments(self, async_client):
        """Test running multiple environment instances concurrently."""
        env_ids = []

        # Create 3 environment instances
        for i in range(3):
            response = async_client.post(
                "/env/Sokoban/initialize", json={"initial_state": SIMPLE_SOKOBAN}
            )
            assert response.status_code == 200
            env_ids.append(response.json()["env_id"])

        # Verify all instances exist
        assert len(instances) == 3

        # Step each instance
        for env_id in env_ids:
            response = async_client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": env_id,
                    "request_id": f"test-multi-{env_id[:8]}",
                    "action": {
                        "tool_calls": [{"tool": "interact", "args": {"action": 0}}]
                    },
                },
            )
            assert response.status_code == 200

        # Terminate all instances
        for env_id in env_ids:
            response = async_client.post(
                "/env/Sokoban/terminate", json={"env_id": env_id}
            )
            assert response.status_code == 200

        # Verify all instances removed
        assert len(instances) == 0

    def test_legacy_endpoints_deprecated(self, client):
        """Test that legacy endpoints are marked as deprecated."""
        # The OpenAPI schema should mark them as deprecated
        response = client.get("/openapi.json")
        openapi = response.json()

        # Check deprecated endpoints
        deprecated_paths = [
            "/env/{env_type}/create",
            "/env/{env_type}/{instance_id}/reset",
            "/env/{env_type}/{instance_id}/step",
            "/env/{env_type}/{instance_id}/terminate",
        ]

        for path in deprecated_paths:
            if path in openapi["paths"]:
                for method in openapi["paths"][path].values():
                    if isinstance(method, dict):
                        assert method.get("deprecated", False) is True

    
    async def test_external_environment_loading(self, async_client):
        """Test that external environments can be loaded via config."""
        # This tests the mechanism without actually loading external envs
        with patch.dict(
            "os.environ",
            {
                "EXTERNAL_ENVIRONMENTS": json.dumps(
                    {
                        "external_environments": [
                            {"module": "fake_module", "function": "register"}
                        ]
                    }
                )
            },
        ):
            # The startup event would normally run, but we're just testing
            # that the mechanism exists and doesn't crash
            from service.external_registry import ExternalRegistryConfig

            config = ExternalRegistryConfig(
                external_environments=[{"module": "fake_module"}]
            )
            assert len(config.external_environments) == 1
