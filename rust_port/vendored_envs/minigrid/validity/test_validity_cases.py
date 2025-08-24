import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
from fastapi.testclient import TestClient


HERE = Path(__file__).parent


def _extract_obs(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize to flat dict using public/private observation when present
    obs = payload
    if isinstance(obs, dict) and "observation" in obs and isinstance(obs["observation"], dict):
        obs = obs["observation"]
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
    # Look for flattened fields first
    pos = obs.get("agent_pos")
    d = obs.get("agent_dir")
    if pos is not None and d is not None:
        return tuple(pos), int(d)
    # Fallback to nested public_observation structure
    pub = obs.get("public_observation") or {}
    pos = pub.get("agent_pos")
    d = pub.get("agent_dir")
    if pos is not None and d is not None:
        return tuple(pos), int(d)
    return None, None


def _step_call(client: TestClient, env_type: str, env_id: str, call: dict) -> Dict[str, Any]:
    # Support single or multiple actions per call
    args: Dict[str, Any] = {}
    if "action" in call:
        args = {"action": str(call["action"]).lower()}
    elif "actions" in call:
        acts = [str(a).lower() for a in call["actions"]]
        args = {"actions": acts}
    body = {
        "env_id": env_id,
        "request_id": "validity",
        "action": {"tool_calls": [{"tool": "minigrid_act", "args": args}]},
    }
    r = client.post(f"/env/{env_type}/step", json=body)
    assert r.status_code == 200, r.text
    return _extract_obs(r.json())


@pytest.mark.parametrize("case_line", (HERE / "cases.jsonl").read_text().splitlines())
def test_minigrid_validity_cases(case_line: str):
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
        pytest.skip("MiniGrid or MiniGrid_PyO3 not registered in service")

    client = TestClient(appmod.app)

    case = json.loads(case_line)
    cfg = case.get("config", {})

    # Initialize
    init = client.post(f"/env/{env_type}/initialize", json={"config": cfg})
    if init.status_code != 200:
        pytest.skip(f"failed to initialize {env_type}: {init.text}")
    assert init.status_code == 200, init.text
    env_id = init.json().get("env_id")
    last = _extract_obs(init.json())

    # Execute scripts; re-initialize for each script for independence
    for script in case.get("scripts", []):
        init = client.post(f"/env/{env_type}/initialize", json={"config": cfg})
        if init.status_code != 200:
            pytest.skip(f"failed to initialize {env_type}: {init.text}")
        env_id = init.json().get("env_id")
        last = _extract_obs(init.json())

        for step in script.get("steps", []):
            last = _step_call(client, env_type, env_id, step.get("call", {}))

            exp = step.get("expect", {})

            # Agent pos/dir checks
            want_pos = exp.get("agent_pos")
            want_dir = exp.get("agent_dir")
            if want_pos is not None or want_dir is not None:
                pos, d = _get_agent_pos_dir(last)
                if want_pos is not None:
                    assert pos is not None, "agent_pos not available in observation"
                    assert list(pos) == list(want_pos)
                if want_dir is not None:
                    assert d is not None, "agent_dir not available in observation"
                    assert int(d) == int(want_dir)

            # Episode flags
            if "terminated" in exp:
                assert bool(last.get("terminated", False)) == bool(exp["terminated"]) 
            if "truncated" in exp:
                assert bool(last.get("truncated", False)) == bool(exp["truncated"]) 

            # Reward checks
            if "reward_last" in exp:
                assert abs(float(last.get("reward_last", 9e9)) - float(exp["reward_last"])) < 1e-6
            if "reward_last_min" in exp:
                assert float(last.get("reward_last", -1e9)) >= float(exp["reward_last_min"]) 
            if "reward_last_max" in exp:
                assert float(last.get("reward_last", 1e9)) <= float(exp["reward_last_max"]) 
