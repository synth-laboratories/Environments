import importlib
from typing import Optional, Tuple

import numpy as np
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


def _load_preset_seed(client: TestClient, seed: int):
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
    assert r.status_code == 200, r.text
    env_id = r.json()["env_id"]
    obs0 = _extract_public(r.json())
    return env_id, fixed, state, obs0


def _step_idx(client: TestClient, env_id: str, idx: int):
    body = {
        "env_id": env_id,
        "request_id": "diff",
        "action": {"tool_calls": [{"tool": "interact", "args": {"action": int(idx)}}]},
    }
    r = client.post("/env/Sokoban_PyO3/step", json=body)
    assert r.status_code == 200, r.text
    return _extract_public(r.json())


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

    env_id, fixed, state, obs0 = _load_preset_seed(client, seed)
    row, col = _find_player(state)
    found = _find_empty_move(fixed, state, row, col)
    if not found:
        pytest.skip("no empty-adjacent move available from start for this seed")
    name, dy, dx = found
    move_idx = DIRS[name][3]

    start_x, start_y = tuple(obs0.get("player_position"))  # PyO3 uses (x,y)
    obs1 = _step_idx(client, env_id, move_idx)
    px1, py1 = tuple(obs1.get("player_position"))
    # In PyO3 coords, x += dx, y += dy
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

    env_id, fixed, state, obs0 = _load_preset_seed(client, seed)
    row, col = _find_player(state)
    found = _find_pushable(fixed, state, row, col)
    if not found:
        pytest.skip("no pushable configuration adjacent to start for this seed")
    name, dy, dx = found
    push_idx = DIRS[name][2]

    # Expected boxes_on_target delta
    y1, x1 = row + dy, col + dx
    y2, x2 = row + 2 * dy, col + 2 * dx
    prev_bot = int(obs0.get("boxes_on_target", 0))
    inc = 0
    if state[y1, x1] == 4 and fixed[y2, x2] == 2:
        inc = 1
    if state[y1, x1] == 3 and fixed[y2, x2] != 2:
        inc = -1

    start_x, start_y = tuple(obs0.get("player_position"))
    obs1 = _step_idx(client, env_id, push_idx)
    px1, py1 = tuple(obs1.get("player_position"))
    # Player moves into the box's old location
    assert (px1, py1) == (start_x + dx, start_y + dy)
    # Boxes on target count changes as expected (0, +1, or -1)
    assert int(obs1.get("boxes_on_target", -999)) == prev_bot + inc
