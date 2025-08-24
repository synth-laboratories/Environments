import importlib
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient


HERE = Path(__file__).parent


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


def _step_call(client: TestClient, env_name: str, env_id: str, call: dict) -> dict:
    # call is either {action:int} or {direction:str, mode:str}
    tool_args = call
    req = {
        "env_id": env_id,
        "request_id": "validity",
        "action": {"tool_calls": [{"tool": "interact", "args": tool_args}]},
    }
    r = client.post(f"/env/{env_name}/step", json=req)
    assert r.status_code == 200, r.text
    return _extract_public(r.json())


@pytest.mark.parametrize("case_line", (HERE / "cases.jsonl").read_text().splitlines())
def test_sokoban_validity_cases_pyo3(case_line):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")

    regs = importlib.import_module("horizons.environments.service.registry")
    if "Sokoban_PyO3" not in regs.list_supported_env_types():
        pytest.skip("Sokoban_PyO3 not registered (horizons_env_py missing?)")

    client = TestClient(appmod.app)

    case = json.loads(case_line)
    cfg = case["config"]
    init = client.post("/env/Sokoban_PyO3/initialize", json={"config": cfg})
    assert init.status_code == 200, init.text
    env_id = init.json()["env_id"]
    init_pub = _extract_public(init.json())

    # Sanity of initial map
    assert isinstance(init_pub.get("room_text"), str) and len(init_pub["room_text"]) > 0
    assert int(init_pub.get("num_boxes", 0)) >= 0
    assert int(init_pub.get("max_steps", 0)) > 0

    # Execute scripts
    for script in case.get("scripts", []):
        # Re-initialize for each script to ensure independence
        init = client.post("/env/Sokoban_PyO3/initialize", json={"config": cfg})
        assert init.status_code == 200
        env_id = init.json()["env_id"]

        last = _extract_public(init.json())
        for step in script.get("steps", []):
            last = _step_call(client, "Sokoban_PyO3", env_id, step["call"]) 
            # num_env_steps must increment by 1
            assert int(last.get("num_env_steps", 0)) >= 1
            # room_fixed immutability heuristic: count walls
            assert init_pub.get("room_text").count("#") == last.get("room_text").count("#")

            exp = step.get("expect", {})
            if "boxes_on_target" in exp:
                assert int(last.get("boxes_on_target", -1)) == int(exp["boxes_on_target"])  # exact match
            if "terminated" in exp:
                assert bool(last.get("terminated", False)) == bool(exp["terminated"]) 
            if "truncated" in exp:
                assert bool(last.get("truncated", False)) == bool(exp["truncated"]) 
            if "reward_last" in exp:
                assert abs(float(last.get("reward_last", 9e9)) - float(exp["reward_last"])) < 1e-6
            if "reward_last_min" in exp:
                assert float(last.get("reward_last", 0.0)) >= float(exp["reward_last_min"]) 
            if "room_text_contains" in exp:
                assert exp["room_text_contains"] in last.get("room_text", "")


# ============ Parity preset tests (migrated) ============

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
    def d1(a):
        tbl = {0: "up", 1: "down", 2: "left", 3: "right"}
        return tbl.get((int(a) - 1) % 4)

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
    pure_id = r_pure.json()["env_id"]
    pure0 = _extract_public(r_pure.json())

    r_rust = client.post(
        "/env/Sokoban_PyO3/initialize",
        json={"config": {"room_fixed": snap["room_fixed"], "room_state": snap["room_state"], "max_steps": snap.get("max_steps", 120), "num_boxes": snap.get("num_boxes", 1)}},
    )
    rust_id = r_rust.json()["env_id"]
    rust0 = _extract_public(r_rust.json())

    assert pure0["room_text"].strip() == rust0["room_text"].strip()

    p1 = _step(client, "Sokoban", pure_id, 0)
    r1 = _step(client, "Sokoban_PyO3", rust_id, 0)
    assert p1["room_text"].strip() == r1["room_text"].strip()
    assert p1["room_text"].strip() == pure0["room_text"].strip()
    assert r1["room_text"].strip() == rust0["room_text"].strip()


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
        json={"config": {"room_fixed": snap["room_fixed"], "room_state": snap["room_state"], "max_steps": snap.get("max_steps", 120), "num_boxes": snap.get("num_boxes", 1)}},
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


# ============ Action-diff checks (migrated) ============

DIRS = {
    "up": (-1, 0, 1, 5),     # dy, dx, push_idx, move_idx
    "down": (1, 0, 2, 6),
    "left": (0, -1, 3, 7),
    "right": (0, 1, 4, 8),
}


def _find_player(state: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(state == 5)
    assert len(ys) == 1
    return int(ys[0]), int(xs[0])


def _load_preset_seed_pyo3(client: TestClient, seed: int):
    pl = importlib.import_module("horizons.examples.sokoban.puzzle_loader")
    pz = pl.get_puzzle_by_seed("easy", seed)
    if pz is None:
        pytest.skip(f"no puzzle for seed {seed}")
    snap = pz.to_engine_snapshot()
    fixed = np.asarray(snap["room_fixed"], dtype=int)
    state = np.asarray(snap["room_state"], dtype=int)
    py_cfg = {
        "room_fixed": snap["room_fixed"],
        "room_state": snap["room_state"],
        "max_steps": snap.get("max_steps", 120),
        "num_boxes": snap.get("num_boxes", 1),
    }
    r = client.post("/env/Sokoban_PyO3/initialize", json={"config": py_cfg})
    env_id = r.json()["env_id"]
    obs0 = _extract_public(r.json())
    return env_id, fixed, state, obs0


def _find_empty_move(fixed: np.ndarray, state: np.ndarray, row: int, col: int) -> Optional[Tuple[str, int, int]]:
    for name, (dy, dx, push_idx, move_idx) in DIRS.items():
        y1, x1 = row + dy, col + dx
        if 0 <= y1 < fixed.shape[0] and 0 <= x1 < fixed.shape[1]:
            if fixed[y1, x1] != 0 and state[y1, x1] in (1, 2):
                return name, dy, dx
    return None


def _find_pushable(fixed: np.ndarray, state: np.ndarray, row: int, col: int) -> Optional[Tuple[str, int, int]]:
    for name, (dy, dx, push_idx, move_idx) in DIRS.items():
        y1, x1 = row + dy, col + dx
        y2, x2 = row + 2 * dy, col + 2 * dx
        if 0 <= y1 < fixed.shape[0] and 0 <= x1 < fixed.shape[1] and 0 <= y2 < fixed.shape[0] and 0 <= x2 < fixed.shape[1]:
            if fixed[y1, x1] != 0 and (state[y1, x1] in (3, 4)) and fixed[y2, x2] != 0 and state[y2, x2] in (1, 2):
                return name, dy, dx
    return None


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_move_action_diff_updates_player_as_expected(seed):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    regs = importlib.import_module("horizons.environments.service.registry")
    if "Sokoban_PyO3" not in regs.list_supported_env_types():
        pytest.skip("Sokoban_PyO3 not registered (horizons_env_py missing?)")
    client = TestClient(appmod.app)

    env_id, fixed, state, obs0 = _load_preset_seed_pyo3(client, seed)
    row, col = _find_player(state)
    found = _find_empty_move(fixed, state, row, col)
    if not found:
        pytest.skip("no empty-adjacent move available from start for this seed")
    name, dy, dx = found
    move_idx = DIRS[name][3]

    start_x, start_y = tuple(obs0.get("player_position"))
    obs1 = _step(client, "Sokoban_PyO3", env_id, move_idx)
    px1, py1 = tuple(obs1.get("player_position"))
    assert (px1, py1) == (start_x + dx, start_y + dy)
    assert int(obs1.get("boxes_on_target", -1)) == int(obs0.get("boxes_on_target", -1))
    assert obs1.get("terminated") is False
    assert obs1.get("reward_last", 0.0) <= 0.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_push_action_diff_moves_box_and_player(seed):
    try:
        appmod = importlib.import_module("horizons.environments.service.app")
    except Exception as e:
        pytest.skip(f"service app not importable: {e}")
    regs = importlib.import_module("horizons.environments.service.registry")
    if "Sokoban_PyO3" not in regs.list_supported_env_types():
        pytest.skip("Sokoban_PyO3 not registered (horizons_env_py missing?)")
    client = TestClient(appmod.app)

    env_id, fixed, state, obs0 = _load_preset_seed_pyo3(client, seed)
    row, col = _find_player(state)
    found = _find_pushable(fixed, state, row, col)
    if not found:
        pytest.skip("no pushable configuration adjacent to start for this seed")
    name, dy, dx = found
    push_idx = DIRS[name][2]

    y1, x1 = row + dy, col + dx
    y2, x2 = row + 2 * dy, col + 2 * dx
    prev_bot = int(obs0.get("boxes_on_target", 0))
    inc = 0
    if state[y1, x1] == 4 and fixed[y2, x2] == 2:
        inc = 1
    if state[y1, x1] == 3 and fixed[y2, x2] != 2:
        inc = -1

    start_x, start_y = tuple(obs0.get("player_position"))
    obs1 = _step(client, "Sokoban_PyO3", env_id, push_idx)
    px1, py1 = tuple(obs1.get("player_position"))
    assert (px1, py1) == (start_x + dx, start_y + dy)
    assert int(obs1.get("boxes_on_target", -999)) == prev_bot + inc
