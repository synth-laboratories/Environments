#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Dict, List

from fastapi.testclient import TestClient


def load_puzzle(difficulty: str, seed: int):
    pl = importlib.import_module("horizons.examples.sokoban.puzzle_loader")
    p = pl.get_puzzle_by_seed(difficulty, seed)
    if p is None:
        raise SystemExit(f"No puzzle for difficulty={difficulty} seed={seed}")
    return p


def build_app_client():
    appmod = importlib.import_module("horizons.environments.service.app")
    return TestClient(appmod.app)


def init_env(client: TestClient, env_name: str, snap: Dict[str, Any]):
    if env_name == "Sokoban_PyO3":
        cfg = {
            "room_fixed": snap["room_fixed"],
            "room_state": snap["room_state"],
            "max_steps": snap.get("max_steps", 120),
            "num_boxes": snap.get("num_boxes", 1),
        }
        r = client.post("/env/Sokoban_PyO3/initialize", json={"config": cfg})
    else:
        r = client.post("/env/Sokoban/initialize", json={"initial_state": snap})
    r.raise_for_status()
    return r.json()["env_id"], r.json()


def extract_public(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "observation" in payload:
        payload = payload["observation"]
    if "public_observation" in payload:
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


def run_solution(seed: int, difficulty: str, env_name: str, use_auto: bool = True) -> int:
    client = build_app_client()
    puzzle = load_puzzle(difficulty, seed)
    snap = puzzle.to_engine_snapshot()
    env_id, init_payload = init_env(client, env_name, snap)
    pub = extract_public(init_payload)
    print("INIT: steps=0 boxes=", pub.get("boxes_on_target"), "/", pub.get("num_boxes"))
    print(pub.get("room_text", ""))

    solution: List[int] = puzzle.solution_path
    total_reward = float(pub.get("total_reward", 0.0) or 0.0)
    def to_dir_mode(a: int):
        # Map 0..3 to Right,Up,Down,Left as direction-only; 4..7 to Move Right,Up,Down,Left
        # This matches our Rust env mapping; when use_auto=True, 0..3 will be sent as mode=auto.
        dirmap = {0: "right", 1: "up", 2: "down", 3: "left"}
        if a in dirmap:
            return dirmap[a], ("auto" if use_auto else "push")
        movemap = {4: "right", 5: "up", 6: "down", 7: "left"}
        if a in movemap:
            return movemap[a], "move"
        # Fallback: treat as direction-only right
        return "right", ("auto" if use_auto else "push")

    for i, act in enumerate(solution, start=1):
        tool_args: Dict[str, Any]
        if env_name.endswith("PyO3"):
            # Prefer direction+mode to leverage Auto in Rust
            dir_s, mode_s = to_dir_mode(int(act))
            tool_args = {"direction": dir_s, "mode": mode_s}
        else:
            tool_args = {"action": int(act)}

        body = {"env_id": env_id, "request_id": f"step-{i}", "action": {"tool_calls": [{"tool": "interact", "args": tool_args}]}}
        r = client.post(f"/env/{env_name}/step", json=body)
        r.raise_for_status()
        pub = extract_public(r.json())
        total_reward = float(pub.get("total_reward", total_reward) or total_reward)
        print(f"step {i:03d} a={act} r_last={pub.get('reward_last')} total={total_reward}")
        if pub.get("terminated") or pub.get("truncated"):
            print("TERMINATED=" , pub.get("terminated"), " TRUNCATED=", pub.get("truncated"))
            print(pub.get("room_text", ""))
            # Basic solved check
            solved = (pub.get("boxes_on_target") == pub.get("num_boxes")) and pub.get("terminated")
            print("SOLVED:", solved)
            return 0 if solved else 2

    # If solution did not terminate, it's unexpected
    print("Solution did not terminate the environment")
    return 3


def main():
    ap = argparse.ArgumentParser(description="Run a Sokoban solution against the service.")
    ap.add_argument("--seed", type=int, required=True, help="Seed to select puzzle deterministically")
    ap.add_argument("--difficulty", type=str, default="easy", help="Difficulty: easy/ultra_easy/...")
    ap.add_argument("--env", type=str, default="Sokoban_PyO3", choices=["Sokoban_PyO3", "Sokoban"], help="Environment to run")
    ap.add_argument("--no-auto", action="store_true", help="Disable auto mode for 0..3 (use push-only)")
    args = ap.parse_args()
    sys.exit(run_solution(args.seed, args.difficulty, args.env, use_auto=(not args.no_auto)))


if __name__ == "__main__":
    main()
