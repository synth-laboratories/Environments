# tictactoe_rs (Rust)

Pure TicTacToe logic crate with:
- Minimal board representation and helpers
- Game-theoretic optimal minimax (`best_response`)
- Deterministic RNG and blended policy (mix minimax with random)

Example
```rust
use tictactoe_rs::{Board, Mark, best_response, LcgRng, blended_move_with_rng};

let b = Board::empty();
let x_best = best_response(&b, Mark::X).unwrap();

let mut rng = LcgRng::new(123);
let o_blended = blended_move_with_rng(&b, Mark::O, 0.5, &mut rng).unwrap();
```

Tests
```
cargo test -p tictactoe-rs
```
