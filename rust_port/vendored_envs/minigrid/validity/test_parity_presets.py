import importlib
import pytest
from fastapi.testclient import TestClient


def _extract_obs(payload):
    return payload.get("observation", payload)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mini_parity_scaffold(seed: int):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
        regs = importlib.import_module("horizons.environments.service.registry")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")

    # Python MiniGrid must exist
    if "MiniGrid" not in regs.list_supported_env_types():
        pytest.skip("MiniGrid not registered in service")

    # Rust bridge is optional for now; skip if absent
    if "MiniGrid_PyO3" not in regs.list_supported_env_types():
        pytest.skip("MiniGrid_PyO3 not registered; parity will run once Rust bridge is ready")

    client = TestClient(appmod.app)
    cfg = {"env_name": "MiniGrid-Empty-5x5-v0", "seed": int(seed)}

    # Python env
    py_init = client.post("/env/MiniGrid/initialize", json={"config": cfg})
    assert py_init.status_code == 200
    py_id = py_init.json()["env_id"]
    py0 = _extract_obs(py_init.json())

    # Rust bridge env (interface to match Python service)
    rs_init = client.post("/env/MiniGrid_PyO3/initialize", json={"config": cfg})
    assert rs_init.status_code == 200
    rs_id = rs_init.json()["env_id"]
    rs0 = _extract_obs(rs_init.json())

    # Basic invariant: both initial observations exist; deeper parity to be added alongside Rust impl
    assert py_id and rs_id and py0 and rs0

