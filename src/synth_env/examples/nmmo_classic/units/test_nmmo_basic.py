"""Basic tests for Neural MMO Classic Environment structure."""

import pytest
import numpy as np
from dataclasses import asdict

# Test the dataclasses and basic structure without requiring NMMO dependencies
try:
    from synth_env.examples.nmmo_classic.engine import (
        NeuralMMOPublicState,
        NeuralMMOPrivateState,
        NeuralMMOEngineSnapshot,
        SkillSnapshot,
    )

    NMMO_AVAILABLE = True
except ImportError:
    NMMO_AVAILABLE = False


@pytest.mark.skipif(not NMMO_AVAILABLE, reason="NMMO dependencies not available")
class TestNeuralMMODataStructures:
    """Test the data structures for Neural MMO."""

    def test_public_state_creation(self):
        """Test creation of NeuralMMOPublicState."""
        state = NeuralMMOPublicState(
            tick=42,
            num_steps_taken=42,
            max_episode_steps=1000,
            agent_id=1,
            position=(50, 50),
            facing=0,
            health=100,
            stamina=100,
            inventory={"food": 10, "water": 5},
            local_terrain=np.zeros((15, 15), dtype=np.int16),
            visible_entities=np.array([[1, 2, 3]]),
            team_score=0.0,
            personal_score=0.0,
        )

        assert state.tick == 42
        assert state.position == (50, 50)
        assert state.health == 100
        assert isinstance(state.local_terrain, np.ndarray)
        assert state.local_terrain.shape == (15, 15)

    def test_private_state_creation(self):
        """Test creation of NeuralMMOPrivateState."""
        skills = {
            "melee": SkillSnapshot(lvl=1, xp=0),
            "range": SkillSnapshot(lvl=1, xp=0),
        }

        state = NeuralMMOPrivateState(
            reward_last_step=1.0,
            total_reward_episode=5.0,
            terminated=False,
            truncated=False,
            env_rng_state_snapshot={},
            skills=skills,
            achievements_status={"first_kill": True},
            agent_internal_stats={"hp": 100},
        )

        assert state.reward_last_step == 1.0
        assert state.total_reward_episode == 5.0
        assert not state.terminated
        assert not state.truncated
        assert len(state.skills) == 2
        assert state.skills["melee"].lvl == 1

    def test_public_state_diff(self):
        """Test diff functionality between public states."""
        state1 = NeuralMMOPublicState(
            tick=1,
            num_steps_taken=1,
            max_episode_steps=1000,
            agent_id=1,
            position=(50, 50),
            facing=0,
            health=100,
            stamina=100,
            inventory={"food": 10},
            local_terrain=np.zeros((5, 5)),
            visible_entities=np.array([[1, 2, 3]]),
            team_score=0.0,
            personal_score=0.0,
        )

        state2 = NeuralMMOPublicState(
            tick=2,
            num_steps_taken=2,
            max_episode_steps=1000,
            agent_id=1,
            position=(51, 50),
            facing=1,
            health=95,
            stamina=100,
            inventory={"food": 10},
            local_terrain=np.zeros((5, 5)),
            visible_entities=np.array([[1, 2, 3]]),
            team_score=0.0,
            personal_score=0.0,
        )

        diff = state2.diff(state1)

        assert "tick" in diff
        assert diff["tick"] == (1, 2)
        assert "position" in diff
        assert diff["position"] == ((50, 50), (51, 50))
        assert "health" in diff
        assert diff["health"] == (100, 95)

    def test_private_state_diff(self):
        """Test diff functionality between private states."""
        state1 = NeuralMMOPrivateState(
            reward_last_step=1.0,
            total_reward_episode=5.0,
            terminated=False,
            truncated=False,
            env_rng_state_snapshot={},
            skills={"melee": SkillSnapshot(lvl=1, xp=0)},
            achievements_status={"first_kill": False},
            agent_internal_stats={"hp": 100},
        )

        state2 = NeuralMMOPrivateState(
            reward_last_step=2.0,
            total_reward_episode=7.0,
            terminated=False,
            truncated=False,
            env_rng_state_snapshot={},
            skills={"melee": SkillSnapshot(lvl=1, xp=10)},
            achievements_status={"first_kill": True},
            agent_internal_stats={"hp": 95},
        )

        diff = state2.diff(state1)

        assert "reward_last_step" in diff
        assert diff["reward_last_step"] == (1.0, 2.0)
        assert "total_reward_episode" in diff
        assert diff["total_reward_episode"] == (5.0, 7.0)
        assert "achievements_status" in diff

    def test_skill_snapshot(self):
        """Test SkillSnapshot dataclass."""
        skill = SkillSnapshot(lvl=5, xp=1000)

        assert skill.lvl == 5
        assert skill.xp == 1000

        # Test that it can be converted to dict
        skill_dict = asdict(skill)
        assert skill_dict == {"lvl": 5, "xp": 1000}

    def test_engine_snapshot(self):
        """Test NeuralMMOEngineSnapshot dataclass."""
        snapshot = NeuralMMOEngineSnapshot(
            task_instance_dict={"id": "test"},
            pickle_blob=b"test_data",
            total_reward=42.0,
            controlled_agent_id=1,
            rng_state={"bit_generator": "test"},
        )

        assert snapshot.total_reward == 42.0
        assert snapshot.controlled_agent_id == 1
        assert snapshot.pickle_blob == b"test_data"


class TestNeuralMMOStructureStandalone:
    """Tests that don't require NMMO imports."""

    def test_module_structure_exists(self):
        """Test that the module files exist with expected structure."""
        from pathlib import Path

        base_path = Path(__file__).parent.parent

        # Check that key files exist
        assert (base_path / "engine.py").exists()
        assert (base_path / "environment.py").exists()
        assert (base_path / "taskset.py").exists()
        assert (base_path / "pyproject.toml").exists()

        # Check maps directory exists
        assert (base_path / "maps" / "medium" / "map1").exists()

    def test_imports_work_when_dependencies_available(self):
        """Test that imports work if dependencies are available."""
        try:
            from synth_env.examples.nmmo_classic.environment import NeuralMMOEnvironment
            from synth_env.examples.nmmo_classic.taskset import NeuralMMOTaskInstance

            assert True  # If we get here, imports worked
        except ImportError as e:
            # Expected if NMMO dependencies not installed
            assert "nmmo" in str(e).lower() or "pufferlib" in str(e).lower()


if __name__ == "__main__":
    # Simple test runner
    print("Testing Neural MMO Classic structure...")

    # Test that module files exist
    from pathlib import Path

    base_path = Path(__file__).parent.parent
    files_to_check = ["engine.py", "environment.py", "taskset.py", "pyproject.toml"]

    for file in files_to_check:
        if (base_path / file).exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")

    # Check maps
    if (base_path / "maps" / "medium" / "map1").exists():
        print("✓ Maps directory exists")
    else:
        print("✗ Maps directory missing")

    print("Basic structure test completed.")
