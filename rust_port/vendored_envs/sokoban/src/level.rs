use serde::{Deserialize, Serialize};

use crate::board::{Board, Tile};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Level {
    pub width: usize,
    pub height: usize,
    pub room_fixed: Vec<u8>,
    pub room_state: Vec<u8>,
}

impl Level {
    pub fn to_board(&self) -> Board {
        let mut b = Board::new(self.width, self.height);
        b.room_fixed.clone_from(&self.room_fixed);
        b.room_state.clone_from(&self.room_state);
        b
    }
}

/// A tiny static level useful for tests/examples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleLevel;

impl SimpleLevel {
    /// Creates a 5x5 room:
    /// #####
    /// #_O_#
    /// #_X_#
    /// #_P_#
    /// #####
    pub fn build() -> Level {
        let width = 5usize; let height = 5usize; let n = width*height;
        let mut room_fixed = vec![Tile::Empty.code(); n];
        let mut room_state = vec![Tile::Empty.code(); n];
        let idx = |x: usize, y: usize| y*width + x;
        // Walls boundary
        for x in 0..width { room_fixed[idx(x,0)] = Tile::Wall.code(); room_fixed[idx(x,height-1)] = Tile::Wall.code(); }
        for y in 0..height { room_fixed[idx(0,y)] = Tile::Wall.code(); room_fixed[idx(width-1,y)] = Tile::Wall.code(); }
        // Target at (2,1)
        room_fixed[idx(2,1)] = Tile::Target.code();
        // Box at (2,2)
        room_state[idx(2,2)] = Tile::Box.code();
        // Player at (2,3)
        room_state[idx(2,3)] = Tile::Player.code();
        Level { width, height, room_fixed, room_state }
    }
}

/// Tiny deterministic RNG to support seed-based level generation without external deps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LcgRng { state: u64 }

impl LcgRng {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
    #[inline]
    fn step(&mut self) { self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); }
    pub fn next_u32(&mut self) -> u32 { self.step(); (self.state >> 32) as u32 }
    pub fn gen_range(&mut self, upper: usize) -> usize { if upper==0 {0} else {(self.next_u32() as usize) % upper} }
}

impl Level {
    /// Deterministically generate a simple level with borders, one player, N boxes and N targets
    /// placed in free cells chosen by the RNG. Ensures player does not overlap boxes/targets.
    pub fn from_seed(width: usize, height: usize, num_boxes: usize, seed: u64) -> Level {
        let n = width*height;
        let mut room_fixed = vec![Tile::Empty.code(); n];
        let mut room_state = vec![Tile::Empty.code(); n];
        let idx = |x: usize, y: usize| y*width + x;
        // Walls boundary
        for x in 0..width { room_fixed[idx(x,0)] = Tile::Wall.code(); room_fixed[idx(x,height-1)] = Tile::Wall.code(); }
        for y in 0..height { room_fixed[idx(0,y)] = Tile::Wall.code(); room_fixed[idx(width-1,y)] = Tile::Wall.code(); }
        // Collect free inner cells
        let mut free: Vec<(usize,usize)> = Vec::new();
        for y in 1..height-1 { for x in 1..width-1 { free.push((x,y)); } }
        let mut rng = LcgRng::new(seed);
        // Place targets
        let mut targets: Vec<(usize,usize)> = Vec::new();
        for _ in 0..num_boxes { if free.is_empty() { break; } let i = rng.gen_range(free.len()); let (x,y) = free.swap_remove(i); room_fixed[idx(x,y)] = Tile::Target.code(); targets.push((x,y)); }
        // Place boxes (avoid targets so we can track on-target separately)
        let mut boxes: Vec<(usize,usize)> = Vec::new();
        for _ in 0..num_boxes { if free.is_empty() { break; } let i = rng.gen_range(free.len()); let (x,y) = free.swap_remove(i); room_state[idx(x,y)] = Tile::Box.code(); boxes.push((x,y)); }
        // Place player
        if !free.is_empty() { let i = rng.gen_range(free.len()); let (x,y) = free.swap_remove(i); room_state[idx(x,y)] = Tile::Player.code(); }
        Level { width, height, room_fixed, room_state }
    }
}
