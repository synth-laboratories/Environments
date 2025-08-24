use serde::{Deserialize, Serialize};

use crate::board::{Board, Tile};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction { Up, Down, Left, Right }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// No operation (gym action=0). Still counts a step and applies per-step reward.
    Noop,
    /// Explicit push attempt; only succeeds if a box is adjacent and the destination is free.
    Push(Direction),
    /// Explicit move attempt; only succeeds if the next cell is free of walls/boxes.
    Move(Direction),
    /// Direction-only semantics (gym-style): try to push, otherwise move if possible.
    Auto(Direction),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameConfig {
    pub width: usize,
    pub height: usize,
    pub max_steps: u32,
}

impl Default for GameConfig {
    fn default() -> Self { Self { width: 7, height: 7, max_steps: 120 } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub player_pos: (usize, usize),
    pub num_env_steps: u32,
    pub max_steps: u32,
    pub reward_last: f64,
    pub total_reward: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepOutcome {
    pub terminated: bool,
    pub truncated: bool,
    pub moved: bool,
    pub pushed_box: bool,
}

impl GameState {
    pub fn new_from_board(board: Board, player_pos: (usize, usize), max_steps: u32) -> Self {
        Self { board, player_pos, num_env_steps: 0, max_steps, reward_last: 0.0, total_reward: 0.0 }
    }

    pub fn from_level(level: &crate::level::Level, max_steps: u32) -> Self {
        let mut b = Board::new(level.width, level.height);
        // Copy fixed and state layers
        b.room_fixed.clone_from(&level.room_fixed);
        b.room_state.clone_from(&level.room_state);
        let player_pos = b.find_player().unwrap_or((0,0));
        Self::new_from_board(b, player_pos, max_steps)
    }

    #[inline]
    fn dir_delta(dir: Direction) -> (isize, isize) {
        match dir { Direction::Up => (0,-1), Direction::Down => (0,1), Direction::Left => (-1,0), Direction::Right => (1,0) }
    }

    /// Attempt to move the player by one cell (no pushing). Returns true if moved.
    fn try_move(&mut self, dir: Direction) -> bool {
        let (dx, dy) = Self::dir_delta(dir);
        let (x, y) = self.player_pos;
        let nx = x as isize + dx; let ny = y as isize + dy;
        if !self.board.in_bounds(nx, ny) { return false; }
        let nxu = nx as usize; let nyu = ny as usize;
        // Blocked by wall or box
        let fixed = self.board.get_fixed(nxu, nyu);
        let state = self.board.get_state(nxu, nyu);
        if fixed == Tile::Wall { return false; }
        if state == Tile::Box || state == Tile::BoxOnTarget { return false; }
        // Move player: clear old cell, set new cell
        self.board.set_state(x, y, Tile::Empty);
        self.board.set_state(nxu, nyu, Tile::Player);
        self.player_pos = (nxu, nyu);
        true
    }

    /// Attempt to push a box in the given direction. Returns (moved_player, pushed).
    fn try_push(&mut self, dir: Direction) -> (bool, bool) {
        let (dx, dy) = Self::dir_delta(dir);
        let (x, y) = self.player_pos;
        let nx = x as isize + dx; let ny = y as isize + dy;
        if !self.board.in_bounds(nx, ny) { return (false, false); }
        let nxu = nx as usize; let nyu = ny as usize;
        let fixed1 = self.board.get_fixed(nxu, nyu);
        let state1 = self.board.get_state(nxu, nyu);
        if fixed1 == Tile::Wall { return (false, false); }
        // If adjacent cell is box, attempt to push
        let is_box = state1 == Tile::Box || state1 == Tile::BoxOnTarget;
        if !is_box { return (false, false); }
        let bx = nx + dx; let by = ny + dy;
        if !self.board.in_bounds(bx, by) { return (false, false); }
        let bxu = bx as usize; let byu = by as usize;
        let fixed2 = self.board.get_fixed(bxu, byu);
        let state2 = self.board.get_state(bxu, byu);
        // Destination must be free of walls/boxes
        if fixed2 == Tile::Wall { return (false, false); }
        if state2 == Tile::Box || state2 == Tile::BoxOnTarget || state2 == Tile::Player { return (false, false); }

        // Compute new tile for box destination depending on whether target
        let dest_box = if fixed2 == Tile::Target { Tile::BoxOnTarget } else { Tile::Box };
        // Compute new tile for vacated box cell depending on whether fixed had a target underneath
        let _vacated = if fixed1 == Tile::Target { Tile::Target } else { Tile::Empty };

        // Move box
        self.board.set_state(bxu, byu, dest_box);
        // Move player into box's old spot
        self.board.set_state(nxu, nyu, Tile::Player);
        // Clear player's old spot
        self.board.set_state(x, y, Tile::Empty);
        // Restore underlying of box old cell
        self.board.set_state(nxu, nyu, Tile::Player);
        // Ensure box source reflects vacated underlying
        self.board.set_state(nxu, nyu, Tile::Player); // already player
        // Note: vacated tile handled when we set player's old spot to Empty; underlying Target remains in fixed
        // However, need to reset the cell the player moved from depending on fixed
        let fixed0 = self.board.get_fixed(x, y);
        let reset0 = if fixed0 == Tile::Target { Tile::Target } else { Tile::Empty };
        self.board.set_state(x, y, reset0);

        self.player_pos = (nxu, nyu);
        (true, true)
    }

    pub fn step(&mut self, action: Action) -> StepOutcome {
        let mut moved = false;
        let mut pushed = false;
        match action {
            Action::Noop => { /* no movement */ }
            Action::Move(d) => { moved = self.try_move(d); }
            Action::Push(d) => {
                // Mirror gym's _push: try to push; if not possible, attempt a simple move.
                let (m, p) = self.try_push(d);
                moved = m; pushed = p;
                if !moved { moved = self.try_move(d); }
            }
            Action::Auto(d) => {
                let (m, p) = self.try_push(d);
                moved = m; pushed = p;
                if !moved { moved = self.try_move(d); }
            }
        }
        self.num_env_steps = self.num_env_steps.saturating_add(1);
        // Reward shaping to align with Python engine:
        // -0.01 per step, +1.0 on solved
        let solved = self.board.boxes_on_target() == self.board.num_boxes();
        let mut r = -0.01f64;
        if solved { r += 1.0; }
        self.reward_last = r;
        self.total_reward += r;
        let terminated = solved;
        let truncated = !solved && self.num_env_steps >= self.max_steps;
        StepOutcome { terminated, truncated, moved, pushed_box: pushed }
    }
}
