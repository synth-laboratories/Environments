import os
import sys
import asyncio
import importlib
import pytest


def _ensure_pyo3_on_path():
    """Add common Rust target dirs to sys.path for local dev builds."""
    here = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "rust_port", "target", "debug"),
        os.path.join(here, "rust_port", "target", "release"),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


@pytest.mark.asyncio
async def test_obs_shape_and_board_text_matches_spec_keys():
    _ensure_pyo3_on_path()

    # Skip test if the native module isn't present
    try:
        import horizons_env_py  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"horizons_env_py not importable: {e}")

    # Import Python reference env and Rust adapter env
    from horizons.environments.examples.pure.tictactoe import TicTacToeEnvironment
    from horizons.environments.examples.pyO3.tictactoe import RustTicTacToeEnvironment
    from horizons.environments.service.core_routes import create_task_instance_for_environment
    from horizons.environments.environment.tools import EnvToolCall

    # Create minimal tasks
    py_task = create_task_instance_for_environment("TicTacToe")
    rs_task = create_task_instance_for_environment(
        "TicTacToe",
        initial_state={"agent_mark": "X", "seed": 7, "opponent_minimax_prob": 0.0},
    )

    py_env = TicTacToeEnvironment(py_task)
    rs_env = RustTicTacToeEnvironment(rs_task)

    py_obs0 = await py_env.initialize()
    rs_obs0 = await rs_env.initialize()

    # Check required public keys per spec and example
    required_keys = {
        "board_text",
        "current_player",
        "move_count",
        "last_move",
        "winner",
        "terminated",
    }

    assert set(required_keys).issubset(set(py_obs0.public_observation.keys()))
    assert set(required_keys).issubset(set(rs_obs0.public_observation.keys()))

    # Validate board_text format (header and 4 lines)
    py_board = py_obs0.public_observation["board_text"]
    rs_board = rs_obs0.public_observation["board_text"]
    assert py_board.splitlines()[0] == "  A B C"
    assert rs_board.splitlines()[0] == "  A B C"
    assert len(py_board.splitlines()) == 4
    assert len(rs_board.splitlines()) == 4

    # Perform one valid move on both; we do not require identical next states,
    # only that formatting and keys remain stable.
    move = EnvToolCall(tool="interact", args={"letter": "A", "number": 1})
    py_obs1 = await py_env.step([move])
    rs_obs1 = await rs_env.step([move])

    for obs in (py_obs1, rs_obs1):
        pub = obs.public_observation
        assert set(required_keys).issubset(set(pub.keys()))
        bt = pub["board_text"]
        assert bt.splitlines()[0] == "  A B C"
        assert len(bt.splitlines()) == 4
