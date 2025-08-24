import importlib
from typing import Any, Dict, Optional, Tuple

import pytest
from fastapi.testclient import TestClient


def _extract_obs(payload: Dict[str, Any]) -> Dict[str, Any]:
    obs = payload
    if isinstance(obs, dict) and "observation" in obs:
        obs = obs["observation"]
    # Flatten service-style {public_observation, private_observation}
    if isinstance(obs, dict) and "public_observation" in obs:
        pub = dict(obs.get("public_observation") or {})
        priv = obs.get("private_observation") or {}
        if "reward_last" in priv and "reward_last" not in pub:
            pub["reward_last"] = priv["reward_last"]
        if "total_reward" in priv and "total_reward" not in pub:
            pub["total_reward"] = priv["total_reward"]
        return pub
    return obs


def _get_agent_pos_dir(obs: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    # Prefer flattened fields
    pos = obs.get("agent_pos")
    d = obs.get("agent_dir")
    if pos is not None and d is not None:
        return tuple(pos), int(d)
    # Fallback to nested
    pub = obs.get("public_observation") or {}
    pos = pub.get("agent_pos")
    d = pub.get("agent_dir")
    if pos is not None and d is not None:
        return tuple(pos), int(d)
    return None, None


def _step(client: TestClient, env_type: str, env_id: str, action: str) -> Dict[str, Any]:
    body = {
        "env_id": env_id,
        "request_id": "diff",
        "action": {"tool_calls": [{"tool": "minigrid_act", "args": {"action": action}}]},
    }
    r = client.post(f"/env/{env_type}/step", json=body)
    assert r.status_code == 200, r.text
    return _extract_obs(r.json())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_diffs(seed: int):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
        regs = importlib.import_module("horizons.environments.service.registry")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    supported = set(regs.list_supported_env_types())
    # Prefer PyO3 only if the extension module is importable
    use_pyo3 = False
    try:
        import importlib.util as _iu
        use_pyo3 = _iu.find_spec("horizons_env_py") is not None
    except Exception:
        use_pyo3 = False
    env_type = "MiniGrid_PyO3" if ("MiniGrid_PyO3" in supported and use_pyo3) else ("MiniGrid" if "MiniGrid" in supported else None)
    if env_type is None:
        pytest.skip("MiniGrid or MiniGrid_PyO3 not registered")

    client = TestClient(appmod.app)
    cfg = {"env_name": "MiniGrid-Empty-5x5-v0", "seed": int(seed)}
    r = client.post(f"/env/{env_type}/initialize", json={"config": cfg})
    if r.status_code != 200:
        pytest.skip(f"failed to initialize {env_type}: {r.text}")
    env_id = r.json()["env_id"]
    obs0 = _extract_obs(r.json())
    pos0, dir0 = _get_agent_pos_dir(obs0)

    # Right turn increments direction mod 4
    obs1 = _step(client, env_type, env_id, "right")
    pos1, dir1 = _get_agent_pos_dir(obs1)
    assert pos1 == pos0
    assert dir1 == (dir0 + 1) % 4

    # Left turn decrements direction mod 4
    obs2 = _step(client, env_type, env_id, "left")
    pos2, dir2 = _get_agent_pos_dir(obs2)
    assert pos2 == pos1
    assert dir2 == dir0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_move_or_block(seed: int):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
        regs = importlib.import_module("horizons.environments.service.registry")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    supported = set(regs.list_supported_env_types())
    use_pyo3 = False
    try:
        import importlib.util as _iu
        use_pyo3 = _iu.find_spec("horizons_env_py") is not None
    except Exception:
        use_pyo3 = False
    env_type = "MiniGrid_PyO3" if ("MiniGrid_PyO3" in supported and use_pyo3) else ("MiniGrid" if "MiniGrid" in supported else None)
    if env_type is None:
        pytest.skip("MiniGrid or MiniGrid_PyO3 not registered")

    client = TestClient(appmod.app)
    cfg = {"env_name": "MiniGrid-Empty-5x5-v0", "seed": int(seed)}
    r = client.post(f"/env/{env_type}/initialize", json={"config": cfg})
    if r.status_code != 200:
        pytest.skip(f"failed to initialize {env_type}: {r.text}")
    env_id = r.json()["env_id"]
    obs0 = _extract_obs(r.json())
    pos0, dir0 = _get_agent_pos_dir(obs0)

    # Attempt to move forward; if blocked, position stays the same
    obs1 = _step(client, env_type, env_id, "forward")
    pos1, dir1 = _get_agent_pos_dir(obs1)
    moved = pos1 != pos0
    if moved:
        # Position should have advanced by the direction unit vector
        dx_dy = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        dx, dy = dx_dy[dir0]
        assert pos1 == (pos0[0] + dx, pos0[1] + dy)
    else:
        assert pos1 == pos0
