# TicTacToe ReAct Demo

Plays TicTacToe against the random opponent policy using `async-openai`.

Requirements
- Rust toolchain
- Network access to fetch crates and call OpenAI
- `OPENAI_API_KEY` set in environment
- Optional: `OPENAI_MODEL` (defaults to `gpt-5-nano`)

Build and run
- `cd rust_port`
- `cargo run -p tictactoe-react-demo`

Behavior
- Initializes the environment with `opponent_minimax_prob=0.0` (pure random).
- At each turn, prints board and legal moves, asks the model to return `{ "index": N }`.
- Validates and applies the move, then the environment opponent replies.
- Continues until terminal state; prints winner.

Notes
- Use `OPENAI_MODEL=gpt-5-nano` (or any available model). The demo expects concise JSON output.
- To switch the opponent to minimax, set up the environment differently in `main.rs`.

