use serde::{Deserialize, Serialize};

/// Tile encodings (match Python engine semantics documented in plan.md)
/// 0: Wall, 1: Empty, 2: Target, 3: BoxOnTarget, 4: Box, 5: Player, 6: Reserved
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Tile {
    Wall = 0,
    Empty = 1,
    Target = 2,
    BoxOnTarget = 3,
    Box = 4,
    Player = 5,
    Reserved = 6,
}

impl Tile {
    pub fn from_code(code: u8) -> Tile {
        match code {
            0 => Tile::Wall,
            1 => Tile::Empty,
            2 => Tile::Target,
            3 => Tile::BoxOnTarget,
            4 => Tile::Box,
            5 => Tile::Player,
            _ => Tile::Reserved,
        }
    }
    pub fn code(self) -> u8 { self as u8 }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    pub width: usize,
    pub height: usize,
    /// Immutable layout (walls and targets). Uses codes 0,1,2.
    pub room_fixed: Vec<u8>,
    /// Mutable layer (player and boxes). Uses codes 3,4,5 or 1 for empty.
    pub room_state: Vec<u8>,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self { width, height, room_fixed: vec![Tile::Empty.code(); n], room_state: vec![Tile::Empty.code(); n] }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize { y * self.width + x }

    #[inline]
    pub fn in_bounds(&self, x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    pub fn get_fixed(&self, x: usize, y: usize) -> Tile { Tile::from_code(self.room_fixed[self.idx(x,y)]) }
    pub fn get_state(&self, x: usize, y: usize) -> Tile { Tile::from_code(self.room_state[self.idx(x,y)]) }
    pub fn set_fixed(&mut self, x: usize, y: usize, t: Tile) { let i = self.idx(x,y); self.room_fixed[i] = t.code(); }
    pub fn set_state(&mut self, x: usize, y: usize, t: Tile) { let i = self.idx(x,y); self.room_state[i] = t.code(); }

    pub fn find_player(&self) -> Option<(usize, usize)> {
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get_state(x,y) == Tile::Player { return Some((x,y)); }
            }
        }
        None
    }

    pub fn boxes_on_target(&self) -> usize {
        self.room_state
            .iter()
            .filter(|&&c| Tile::from_code(c) == Tile::BoxOnTarget)
            .count()
    }

    pub fn num_boxes(&self) -> usize {
        self.room_state
            .iter()
            .filter(|&&c| { let t = Tile::from_code(c); t == Tile::Box || t == Tile::BoxOnTarget })
            .count()
    }

    /// Render a text view matching Python's GRID_LOOKUP with 3-char glyphs per cell.
    pub fn room_text(&self) -> String {
        fn glyph(fixed: Tile, state: Tile) -> &'static str {
            // Mirror GRID_LOOKUP: 0: " # ", 1: " _ ", 2: " O ", 3: " √ ", 4: " X ", 5: " P ", 6: " S "
            // Compose an effective code similar to Python's rendering logic.
            // Priority: walls override; then dynamic entities; then targets; then empty.
            if fixed == Tile::Wall { return " # "; }
            match state {
                Tile::Player => " P ",
                Tile::BoxOnTarget => " √ ",
                Tile::Box => " X ",
                _ => {
                    if fixed == Tile::Target { " O " } else { " _ " }
                }
            }
        }
        let mut out = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let g = glyph(self.get_fixed(x,y), self.get_state(x,y));
                out.push_str(g);
            }
            if y + 1 < self.height { out.push('\n'); }
        }
        out
    }
}
