MiniGrid Validity Suite

Purpose
- Lock in Python MiniGrid semantics (actions, rewards, observations) and provide a co-located test suite used to validate a future Rust port.
- Modeled after the Sokoban validity folder for consistency and ease of running.

What’s covered
- Action mapping: 0..6 → left, right, forward, pickup, drop, toggle, done.
- Rotation/forward behavior and blocking on walls/boundaries/objects.
- Step penalty accounting and total reward accumulation.
- Goal reaching and termination semantics on simple envs.
- Interaction (pickup/drop/toggle) on DoorKey/Unlock variants.
- Observation encoding and deterministic resets by seed+env.

Running
- From repo root: `pytest rust_port/vendored_envs/minigrid/validity -q`
- Or run the helper: `bash rust_port/vendored_envs/minigrid/validity/run_all.sh`
  - Skips gracefully if the service app or MiniGrid environment is unavailable.

Files
- `cases.jsonl`: JSONL scenarios with scripted calls and expected outcomes.
- `test_validity_cases.py`: JSONL runner that drives the service and asserts expectations.
- `test_action_diffs.py`: Direction/forward/interaction delta checks.
- `test_parity_presets.py`: Parity scaffolding for future Rust bridge (skips for now if bridge missing).
- `run_all.sh`: Executes pytest for validity; will later call `cargo test` once Rust crates exist.

Notes
- This suite targets the service API (`/env/MiniGrid/*`) and uses the environment’s tool `minigrid_act`.
- Expectations assert against unified observation fields (public/private) as exposed by the service layer.

