//! Pure Sokoban logic crate.
//! - Board and tile encoding
//! - Game state and movement/push rules
//! - Deterministic serialization with serde

mod board;
mod game;
mod level;
mod preset;

pub use board::{Board, Tile};
pub use game::{Action, Direction, GameConfig, GameState, StepOutcome};
pub use level::{Level, SimpleLevel};
pub use preset::preset_level;
