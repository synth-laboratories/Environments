#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Dict, Any, Tuple

from fastapi.testclient import TestClient

from horizons.environments.service.app import app
from horizons.examples.sokoban.puzzle_loader import get_puzzle_by_seed


def extract_obs(o: Dict[str, Any]) -> Dict[str, Any]:
    # Service wraps as {observation, reward, done, info}
    if "observation" in o:
        o = o["observation"]
    # Handle PyO3 internal observation shape
    if "public_observation" in o:
        pub = o["public_observation"] or {}
        priv = o.get("private_observation") or {}
        # Lift flags for convenience
        pub.setdefault("terminated", bool(priv.get("terminated", pub.get("terminated", False))))
        pub.setdefault("truncated", bool(priv.get("truncated", pub.get("truncated", False))))
        # Lift rewards if present
        pub.setdefault("reward_last", priv.get("reward_last"))
        pub.setdefault("total_reward", priv.get("total_reward"))
        return pub
    # Pure returns flat dict already
    return o


def build_pure_snapshot(seed: int, difficulty: str = "easy") -> Dict[str, Any]:
    p = get_puzzle_by_seed(difficulty, seed)
    if p is None:
        raise RuntimeError(f"No puzzle found for difficulty={difficulty} seed={seed}")
    return p.to_engine_snapshot()


def to_action_index(direction: str, mode: str) -> int:
    d = direction.lower().strip()
    m = mode.lower().strip()
    base = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
    }[d]
    if m == "move":
        return 4 + base
    return base  # push


def step(client: TestClient, env: str, env_id: str, action_idx: int) -> Dict[str, Any]:
    req = {
        "env_id": env_id,
        "request_id": "manual",
        "action": {"tool_calls": [{"tool": "interact", "args": {"action": int(action_idx)}}]},
    }
    r = client.post(f"/env/{env}/step", json=req)
    r.raise_for_status()
    return extract_obs(r.json())


def init_pair(seed: int) -> Tuple[TestClient, str, str, Dict[str, Any], Dict[str, Any]]:
    client = TestClient(app)

    # Pure via snapshot for exact layout
    snap = build_pure_snapshot(seed)
    r_pure = client.post("/env/Sokoban/initialize", json={"initial_state": snap})
    r_pure.raise_for_status()
    pure_id = r_pure.json()["env_id"]
    pure_obs = extract_obs(r_pure.json())

    # PyO3 via seed config; relies on prebaked JSONL for 1:1
    cfg = {"seed": seed, "use_simple_level": False}
    r_rust = client.post("/env/Sokoban_PyO3/initialize", json={"config": cfg})
    r_rust.raise_for_status()
    rust_id = r_rust.json()["env_id"]
    rust_obs = extract_obs(r_rust.json())
    return client, pure_id, rust_id, pure_obs, rust_obs


def show(obs_pure: Dict[str, Any], obs_rust: Dict[str, Any]) -> None:
    pt = obs_pure.get("room_text", "")
    rt = obs_rust.get("room_text", "")
    same = pt.strip() == rt.strip()
    print("\n=== PURE ===\n" + pt)
    print("=== RUST ===\n" + rt)
    print(f"same_map={same}")
    for label, o in [("PURE", obs_pure), ("RUST", obs_rust)]:
        print(
            f"[{label}] steps={o.get('num_env_steps') or o.get('steps_taken')} boxes_on_target={o.get('boxes_on_target')} "
            f"reward_last={o.get('reward_last')} total_reward={o.get('total_reward')} "
            f"terminated={o.get('terminated')} truncated={o.get('truncated')}"
        )


def main():
    seed = 7
    client, pure_id, rust_id, pure_obs, rust_obs = init_pair(seed)
    print(f"Initialized both with seed={seed}")
    show(pure_obs, rust_obs)

    print("\nControls: type e.g. 'push up' or 'move left'. 'q' to quit.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            break
        try:
            if " " in line:
                mode, direction = line.split(None, 1)
            else:
                # default to push if only direction
                direction, mode = line, "push"
            action_idx = to_action_index(direction, mode)
        except Exception as e:
            print(f"bad input: {e}")
            continue
        try:
            pure_obs = step(client, "Sokoban", pure_id, action_idx)
            rust_obs = step(client, "Sokoban_PyO3", rust_id, action_idx)
            show(pure_obs, rust_obs)
            if pure_obs.get("terminated") and rust_obs.get("terminated"):
                print("Both terminated — done.")
                break
        except Exception as e:
            print(f"step error: {e}")
            continue


if __name__ == "__main__":
    main()
