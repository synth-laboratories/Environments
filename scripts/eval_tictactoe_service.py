#!/usr/bin/env python3
"""
Quick eval script: initialize and step both TicTacToe and TicTacToe_PyO3 via the service.

Usage:
  python scripts/eval_tictactoe_service.py --host http://127.0.0.1:8901
"""

import argparse
import json
import sys
import time
from typing import Dict, Any

import requests


def post(host: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{host}{path}"
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def run_eval(host: str, env_name: str) -> None:
    print(f"\n=== Evaluating {env_name} ===")
    init_cfg = {
        "config": {
            "agent_mark": "X",
            "opponent_minimax_prob": 1.0,
            "seed": 7,
        }
    }
    try:
        resp = post(host, f"/env/{env_name}/initialize", init_cfg)
    except requests.HTTPError as e:
        print(f"Initialize failed for {env_name}: {e}")
        sys.exit(1)
    env_id = resp.get("env_id")
    obs = resp.get("observation", {})
    print(f"Initialized {env_name}, env_id={env_id}")
    pub = obs.get("public_observation", {})
    if pub:
        print(pub.get("board_text", "<no board_text>"))
    else:
        # Some impls return raw obs; handle both
        print(obs.get("board_text", "<no board_text>"))

    moves = [("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3)]
    terminated = False
    for i, (letter, number) in enumerate(moves, 1):
        if terminated:
            break
        payload = {
            "env_id": env_id,
            "request_id": f"m{i}",
            "action": {
                "tool_calls": [
                    {"tool": "interact", "args": {"letter": letter, "number": number}}
                ]
            },
        }
        step = post(host, f"/env/{env_name}/step", payload)
        obs = step.get("observation", {})
        pub = obs.get("public_observation", obs)
        print(f"Move {i}: {letter}{number}")
        print(pub.get("board_text", "<no board_text>"))
        terminated = bool(pub.get("terminated", False))
        if terminated:
            print(f"Terminated (winner={pub.get('winner')})")

    # Terminate and summarize
    post(host, f"/env/{env_name}/terminate", {"env_id": env_id})
    print(f"Finished {env_name}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8901")
    args = ap.parse_args()

    # Basic health check
    try:
        r = requests.get(f"{args.host}/health", timeout=5)
        r.raise_for_status()
        print("Health:", r.json())
    except Exception as e:
        print("Service health check failed:", e)
        sys.exit(1)

    for env in ["TicTacToe", "TicTacToe_PyO3"]:
        run_eval(args.host, env)


if __name__ == "__main__":
    main()
