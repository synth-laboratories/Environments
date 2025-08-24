import importlib
import pytest
from fastapi.testclient import TestClient


def _extract_public(payload):
    if isinstance(payload, dict) and "observation" in payload:
        payload = payload["observation"]
    if isinstance(payload, dict) and "public_observation" in payload:
        pub = payload["public_observation"] or {}
        priv = payload.get("private_observation") or {}
        pub.setdefault("terminated", bool(priv.get("terminated", pub.get("terminated", False))))
        pub.setdefault("truncated", bool(priv.get("truncated", pub.get("truncated", False))))
        if "reward_last" not in pub and "reward_last" in priv:
            pub["reward_last"] = priv.get("reward_last")
        if "total_reward" not in pub and "total_reward" in priv:
            pub["total_reward"] = priv.get("total_reward")
        return pub
    return payload


def _step(client: TestClient, env: str, env_id: str, a: int):
    body = {
        "env_id": env_id,
        "request_id": "parity",
        "action": {"tool_calls": [{"tool": "interact", "args": {"action": int(a)}}]},
    }
    r = client.post(f"/env/{env}/step", json=body)
    assert r.status_code == 200, r.text
    return _extract_public(r.json())


def _step_dir_auto(client: TestClient, env: str, env_id: str, direction: str):
    body = {
        "env_id": env_id,
        "request_id": "parity_dir",
        "action": {"tool_calls": [{"tool": "interact", "args": {"direction": direction, "mode": "auto"}}]},
    }
    r = client.post(f"/env/{env}/step", json=body)
    assert r.status_code == 200, r.text
    return _extract_public(r.json())


def _decode_to_directions(sol):
    # Try 1..8 mapping (gym): indices map to directions in repeating blocks of 4
    def d1(a):
        tbl = {0: "up", 1: "down", 2: "left", 3: "right"}
        return tbl.get((int(a) - 1) % 4)

    # Try 0..3 mapping (direction-only) often seen in older logs
    def d2(a):
        tbl = {0: "right", 1: "up", 2: "down", 3: "left"}
        return tbl.get(int(a))

    return [d1(a) for a in sol], [d2(a) for a in sol]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_action_0_noop_is_parity_and_stable_on_presets(seed):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    regs = importlib.import_module("horizons.environments.service.registry")
    if "Sokoban_PyO3" not in regs.list_supported_env_types():
        pytest.skip("Sokoban_PyO3 not registered (horizons_env_py missing?)")
    client = TestClient(appmod.app)
    pl = importlib.import_module("horizons.examples.sokoban.puzzle_loader")

    pz = pl.get_puzzle_by_seed("easy", seed)
    if pz is None:
        pytest.skip(f"no puzzle for seed {seed}")
    snap = pz.to_engine_snapshot()

    r_pure = client.post("/env/Sokoban/initialize", json={"initial_state": snap})
    assert r_pure.status_code == 200
    pure_id = r_pure.json()["env_id"]
    pure0 = _extract_public(r_pure.json())

    py_cfg = {
        "room_fixed": snap["room_fixed"],
        "room_state": snap["room_state"],
        "max_steps": snap.get("max_steps", 120),
        "num_boxes": snap.get("num_boxes", 1),
    }
    r_rust = client.post("/env/Sokoban_PyO3/initialize", json={"config": py_cfg})
    assert r_rust.status_code == 200
    rust_id = r_rust.json()["env_id"]
    rust0 = _extract_public(r_rust.json())

    assert pure0["room_text"].strip() == rust0["room_text"].strip()

    p = _step(client, "Sokoban", pure_id, 0)
    r = _step(client, "Sokoban_PyO3", rust_id, 0)
    assert p["room_text"].strip() == r["room_text"].strip()
    assert p["room_text"].strip() == pure0["room_text"].strip()
    assert r["room_text"].strip() == rust0["room_text"].strip()


def test_step_penalty_accumulates_with_noop_parity():
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    regs = importlib.import_module("horizons.environments.service.registry")
    if "Sokoban_PyO3" not in regs.list_supported_env_types():
        pytest.skip("Sokoban_PyO3 not registered (horizons_env_py missing?)")
    client = TestClient(appmod.app)
    pl = importlib.import_module("horizons.examples.sokoban.puzzle_loader")

    pz = pl.get_puzzle_by_seed("easy", 1)
    if pz is None:
        pytest.skip("no puzzle for seed 1")
    snap = pz.to_engine_snapshot()

    r_pure = client.post("/env/Sokoban/initialize", json={"initial_state": snap})
    pure_id = r_pure.json()["env_id"]
    r_rust = client.post(
        "/env/Sokoban_PyO3/initialize",
        json={"config": {"room_fixed": snap["room_fixed"], "room_state": snap["room_state"], "max_steps": snap.get("max_steps", 120)}},
    )
    rust_id = r_rust.json()["env_id"]

    totals = []
    for _ in range(3):
        p = _step(client, "Sokoban", pure_id, 0)
        r = _step(client, "Sokoban_PyO3", rust_id, 0)
        assert p["room_text"].strip() == r["room_text"].strip()
        assert abs(p.get("reward_last", 0.0) - (-0.01)) < 1e-6
        assert abs(r.get("reward_last", 0.0) - (-0.01)) < 1e-6
        totals.append((p.get("total_reward", 0.0), r.get("total_reward", 0.0)))

    assert all(abs(tp - tr) < 1e-6 for tp, tr in totals)
    assert abs(totals[-1][0] - (-0.03)) < 1e-6
    assert abs(totals[-1][1] - (-0.03)) < 1e-6


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gold_solution_rollout_pyo3_terminates_with_positive_reward(seed):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
        pl = importlib.import_module("horizons.examples.sokoban.puzzle_loader")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    regs = importlib.import_module("horizons.environments.service.registry")
    if "Sokoban_PyO3" not in regs.list_supported_env_types():
        pytest.skip("Sokoban_PyO3 not registered (horizons_env_py missing?)")
    client = TestClient(appmod.app)

    pz = pl.get_puzzle_by_seed("easy", seed)
    if pz is None or not getattr(pz, "solution_path", None):
        pytest.skip(f"no puzzle/solution for seed {seed}")
    snap = pz.to_engine_snapshot()

    py_cfg = {
        "room_fixed": snap["room_fixed"],
        "room_state": snap["room_state"],
        "max_steps": snap.get("max_steps", 120),
        "num_boxes": snap.get("num_boxes", 1),
    }
    r = client.post("/env/Sokoban_PyO3/initialize", json={"config": py_cfg})
    assert r.status_code == 200
    env_id = r.json()["env_id"]

    last = _extract_public(r.json())
    seq1, seq2 = _decode_to_directions(pz.solution_path)

    solved = False
    for seq in (seq1, seq2):
        r = client.post("/env/Sokoban_PyO3/initialize", json={"config": py_cfg})
        env_id = r.json()["env_id"]
        last = _extract_public(r.json())
        for d in seq:
            if not d:
                continue
            last = _step_dir_auto(client, "Sokoban_PyO3", env_id, d)
            if last.get("terminated"):
                solved = True
                break
        if solved:
            break

    assert solved and last.get("terminated") is True
    assert last.get("truncated") is False
    assert int(last.get("boxes_on_target", -1)) == int(last.get("num_boxes", 0))
    assert last.get("reward_last", 0.0) > 0.9
    steps = last.get("num_env_steps") or last.get("steps_taken")
    assert steps is not None and steps > 0
    expected_total = 1.0 - 0.01 * steps
    assert abs(float(last.get("total_reward", 0.0)) - expected_total) < 1e-6

