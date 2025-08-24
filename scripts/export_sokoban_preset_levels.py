"""
Export deterministic Sokoban levels for seeds 1..100 as JSONL for Rust preset loading.

Each line: { seed, dim_room, num_boxes, max_steps, room_fixed, room_state }

Writes to rust_port/vendored_envs/sokoban/data/prebaked_levels.jsonl

Usage:
  . .venv/bin/activate
  python scripts/export_sokoban_preset_levels.py
"""

from __future__ import annotations

import json
from pathlib import Path

from horizons.examples.sokoban.puzzle_loader import get_puzzle_by_seed


OUT = Path("rust_port/vendored_envs/sokoban/data/prebaked_levels.jsonl")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cnt = 0
    with OUT.open("w", encoding="utf-8") as f:
        for seed in range(1, 101):
            # Choose a default difficulty; adjust if you want a different mapping
            puzzle = get_puzzle_by_seed("easy", seed)
            if puzzle is None:
                # Try fallback difficulty
                puzzle = get_puzzle_by_seed("ultra_easy", seed)
            if puzzle is None:
                print(f"warn: no puzzle for seed {seed}")
                continue
            snap = puzzle.to_engine_snapshot()
            row = {
                "seed": seed,
                "dim_room": list(snap.get("dim_room", [puzzle.dim_room[0], puzzle.dim_room[1]])),
                "num_boxes": snap.get("num_boxes", puzzle.num_boxes),
                "max_steps": snap.get("max_steps", puzzle.max_steps),
                "room_fixed": snap["room_fixed"],
                "room_state": snap["room_state"],
            }
            f.write(json.dumps(row))
            f.write("\n")
            cnt += 1
    print(f"Wrote {cnt} presets to {OUT}")


if __name__ == "__main__":
    main()

