# Horizons Rust Port (rust_port)

This workspace hosts the Rust core, vendored environments, and optional bindings/services described in rust_port/rewrite.txt.

Initial targets
- core: common Env trait + types (ToolCall, ToolResult, Observation, Snapshot, EngineError), plus tool/env registries
- envs/tictactoe_env: TicTacToe environment wrapping `vendored_envs/tictactoe_rs` and exposing tools
- vendored_envs/tictactoe_rs: deterministic game logic + policies
- demos/tictactoe_react_demo: ReAct-style agent loop using `async-openai`
- service/env_service: Axum-based HTTP service exposing initialize/step/checkpoint/terminate
- horizons_env_py: minimal PyO3 module to validate the Python import path

Build (local)
- cargo metadata -q  # verify workspace
- cargo test -p tictactoe-env  # env unit tests (requires network for `tokio` if not cached)
- (optional) cargo run -p tictactoe-react-demo  # async-openai demo (requires OPENAI_API_KEY and network)
- (optional) cargo run -p env-service  # HTTP service at 127.0.0.1:8080
- (optional) maturin develop -m horizons_env_py/Cargo.toml  # build and install the PyO3 module into current venv

Service API
- GET `/envs` -> `[String]`: List registered environment types.
- POST `/initialize` -> `{ env_id, observation }`
  - Body: `{ "env_type": "TicTacToe", "config"?: { "agent_mark": "X"|"O", "opponent_minimax_prob": 0.0..1.0, "seed": u64 } }`
- POST `/step` -> `Observation`
  - Body: `{ "env_id": string, "tool_calls": [ { "tool": "place"|"place_coord"|"suggest", "args": {...} } ] }`
- POST `/checkpoint` -> `Snapshot`
  - Body: `{ "env_id": string }`
- POST `/terminate` -> `Observation`
  - Body: `{ "env_id": string }`

Config-aware init
- Core supports `register_environment_with_config(name, factory)` and `create_environment_with_config(name, Option<serde_json::Value>)`.
- Service uses config-aware creation for all envs; TicTacToe registers a factory that parses its `Config` from `config`.

See rust_port/rewrite.txt for the full plan, contracts, and acceptance criteria.
