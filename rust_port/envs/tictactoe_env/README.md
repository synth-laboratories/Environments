# TicTacToe Environment (Rust)

Implements the Horizons `Environment` trait over the vendored `tictactoe-rs` logic.

Tools
- `"place"`: args `{ "index": 0..8 }` — place agent's mark at index.
- `"place_coord"`: args `{ "col": "A"|"B"|"C", "row": 1|2|3 }` — coordinate convenience.
- `"suggest"`: args `{}` — returns a `best_response` suggestion in observation `extra.suggested` without mutating state.

Observation `data`
- `board_text`: fixed-format board rendering.
- `to_move`: `"X"|"O"`.
- `legal_moves`: array of legal indices.
- `winner`: `null|"X"|"O"`.
- `extra`: tool-specific payload.

Config
- `agent_mark`: `"X"|"O"` (default `"X"`).
- `opponent_minimax_prob`: float `[0,1]` blend; `0.0` random, `1.0` minimax (default `0.0`).
- `seed`: opponent RNG seed (default `42`).

Tests
- `suggest_matches_minimax`: validates `suggest` matches minimax.
- `random_opponent_differs_by_seed`: checks seed sensitivity for random opponent.
- `opponent_blocks_with_minimax`: ensures optimal blocking when `opponent_minimax_prob=1.0`.

