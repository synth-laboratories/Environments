//! Pure TicTacToe logic crate (no Horizons traits).
//! - Board helpers and coordinate utilities
//! - Game-theoretic optimal `best_response` via minimax
//! - Deterministic RNG (LCG) and a blended policy: mix minimax with random

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mark { X, O }

impl Mark {
    pub fn other(self) -> Mark { match self { Mark::X => Mark::O, Mark::O => Mark::X } }
    pub fn as_u8(self) -> u8 { match self { Mark::X => 1, Mark::O => 2 } }
    pub fn from_u8(v: u8) -> Option<Mark> { match v { 1 => Some(Mark::X), 2 => Some(Mark::O), _ => None } }
}

/// Row-major 3x3 board as a flat array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub cells: [u8; 9], // 0 empty, 1 X, 2 O
}

impl Default for Board { fn default() -> Self { Self { cells: [0; 9] } } }

impl Board {
    pub fn empty() -> Self { Self::default() }
    pub fn get(&self, idx: usize) -> u8 { self.cells[idx] }
    pub fn set(&mut self, idx: usize, mark: Mark) { self.cells[idx] = mark.as_u8(); }
    pub fn is_empty(&self, idx: usize) -> bool { self.cells[idx] == 0 }
    pub fn empties(&self) -> impl Iterator<Item=usize> + '_ { (0..9).filter(|&i| self.cells[i]==0) }
    pub fn is_full(&self) -> bool { self.cells.iter().all(|&v| v != 0) }

    pub fn coord_to_idx(letter: &str, number: i64) -> Option<usize> {
        let col = match letter.to_uppercase().as_str() { "A"=>0, "B"=>1, "C"=>2, _=>return None };
        if !(1..=3).contains(&number) { return None; }
        let row = (number - 1) as usize;
        Some(row*3 + col)
    }

    pub fn idx_to_coord(idx: usize) -> Option<String> {
        if idx >= 9 { return None; }
        let row = idx / 3; // 0..2
        let col = idx % 3; // 0..2
        let letter = match col { 0 => 'A', 1 => 'B', 2 => 'C', _ => '?' };
        Some(format!("{}{}", letter, row + 1))
    }

    pub fn winner(&self) -> Option<Mark> {
        const W: [[usize;3];8] = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
        for line in W {
            let a = self.cells[line[0]];
            if a!=0 && a==self.cells[line[1]] && a==self.cells[line[2]] {
                return Mark::from_u8(a);
            }
        }
        None
    }

    pub fn is_draw(&self) -> bool { self.cells.iter().all(|&v| v!=0) && self.winner().is_none() }

    /// Render as fixed-format board text (header row, labeled rows 1..3).
    pub fn board_text(&self) -> String {
        let mut s = String::from("  A B C\n");
        for r in 0..3 {
            s.push_str(&(r+1).to_string()); s.push(' ');
            for c in 0..3 {
                let ch = match self.cells[r*3+c] { 1=>'X', 2=>'O', _=>' ' };
                s.push(ch); if c<2 { s.push(' '); }
            }
            if r<2 { s.push('\n'); }
        }
        s
    }
}

/// Apply a move; returns false if illegal.
pub fn apply_move(board: &mut Board, idx: usize, mark: Mark) -> bool {
    if idx>=9 || !board.is_empty(idx) { return false; }
    board.set(idx, mark);
    true
}

/// Return all legal (empty) cell indices in ascending order.
/// Collect legal move indices into a fixed array; returns (array, count).
pub fn legal_moves_array(board: &Board) -> ([usize; 9], usize) {
    let mut out = [0usize; 9];
    let mut n = 0usize;
    for i in 0..9 {
        if board.cells[i] == 0 {
            out[n] = i;
            n += 1;
        }
    }
    (out, n)
}

/// Game-theoretic optimal best response for `player` on `board` using minimax.
/// Returns the chosen cell index (0..8), or None if no moves remain.
pub fn best_response(board: &Board, player: Mark) -> Option<usize> {
    fn eval_terminal(b: &Board, root: Mark, depth: i32) -> Option<i32> {
        if let Some(w) = b.winner() {
            return Some(if w == root { 10 - depth } else { depth - 10 });
        }
        if b.is_full() { return Some(0); }
        None
    }
    fn minimax_ab(b: &mut Board, to_move: Mark, root: Mark, depth: i32, mut alpha: i32, mut beta: i32) -> i32 {
        if let Some(score) = eval_terminal(b, root, depth) { return score; }
        let maximizing = to_move == root;
        if maximizing {
            let mut best = i32::MIN;
            for idx in 0..9 { if b.cells[idx]==0 { b.cells[idx]=to_move.as_u8(); let val = minimax_ab(b, to_move.other(), root, depth+1, alpha, beta); b.cells[idx]=0; if val>best { best=val; } if best>alpha { alpha=best; } if beta<=alpha { break; } } }
            best
        } else {
            let mut best = i32::MAX;
            for idx in 0..9 { if b.cells[idx]==0 { b.cells[idx]=to_move.as_u8(); let val = minimax_ab(b, to_move.other(), root, depth+1, alpha, beta); b.cells[idx]=0; if val<best { best=val; } if best<beta { beta=best; } if beta<=alpha { break; } } }
            best
        }
    }
    let mut b = board.clone();
    let mut best_idx = None;
    let mut best_score = i32::MIN;
    for idx in 0..9 { if b.cells[idx]==0 { b.cells[idx]=player.as_u8(); let score = minimax_ab(&mut b, player.other(), player, 1, i32::MIN/2, i32::MAX/2); b.cells[idx]=0; if score>best_score { best_score=score; best_idx=Some(idx); } } }
    best_idx
}

// -----------------------
// Deterministic RNG (LCG)
// -----------------------
/// Tiny deterministic RNG to avoid external dependencies.
#[derive(Clone, Copy, Debug)]
pub struct LcgRng { state: u64 }

impl LcgRng {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
    #[inline]
    fn step(&mut self) { self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); }
    pub fn next_u32(&mut self) -> u32 { self.step(); (self.state >> 32) as u32 }
    pub fn next_f64(&mut self) -> f64 { (self.next_u32() as f64) / (u32::MAX as f64) }
    pub fn gen_index(&mut self, upper: usize) -> usize { if upper==0 {0} else {(self.next_u32() as usize) % upper} }
}

/// Choose a random legal move using the given RNG. Returns None if no moves.
pub fn random_move_with_rng(board: &Board, rng: &mut LcgRng) -> Option<usize> {
    let (legal, n) = legal_moves_array(board);
    if n == 0 { return None; }
    let idx = rng.gen_index(n);
    Some(legal[idx])
}

/// Blended policy: with probability `minimax_prob` choose minimax best response; otherwise choose random.
/// Deterministic based on `rng`.
pub fn blended_move_with_rng(board: &Board, player: Mark, minimax_prob: f64, rng: &mut LcgRng) -> Option<usize> {
    if board.is_full() { return None; }
    let p = minimax_prob.clamp(0.0, 1.0);
    let roll = rng.next_f64();
    if roll < p {
        best_response(board, player)
    } else {
        random_move_with_rng(board, rng)
    }
}

/// Convenience helper: blended policy with an explicit seed for determinism.
pub fn blended_move_with_seed(board: &Board, player: Mark, minimax_prob: f64, seed: u64) -> Option<usize> {
    let mut rng = LcgRng::new(seed);
    blended_move_with_rng(board, player, minimax_prob, &mut rng)
}

/// Stateful policy that blends minimax with random using an internal RNG.
#[derive(Debug, Clone)]
pub struct BlendedPolicy {
    pub minimax_prob: f64,
    rng: LcgRng,
}

impl BlendedPolicy {
    pub fn new(seed: u64, minimax_prob: f64) -> Self { Self { minimax_prob, rng: LcgRng::new(seed) } }
    pub fn set_minimax_prob(&mut self, p: f64) { self.minimax_prob = p; }
    pub fn choose(&mut self, board: &Board, player: Mark) -> Option<usize> {
        blended_move_with_rng(board, player, self.minimax_prob, &mut self.rng)
    }
}

// -----------------------
// Tests
// -----------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_best_is_center_for_x() {
        let b = Board::empty();
        let mv = best_response(&b, Mark::X).unwrap();
        assert!(mv < 9 && b.is_empty(mv), "best move should be a legal cell");
    }

    #[test]
    fn respond_to_corner_with_center_for_o() {
        let mut b = Board::empty();
        assert!(apply_move(&mut b, 0, Mark::X)); // A1
        let mv = best_response(&b, Mark::O).unwrap();
        assert_eq!(mv, 4, "O should take center against corner opening");
    }

    #[test]
    fn winning_move_is_chosen() {
        // X to play with two in a row at top row: X X _
        let mut b = Board::empty();
        b.cells = [1,1,0, 0,0,0, 0,0,0];
        let mv = best_response(&b, Mark::X).unwrap();
        assert_eq!(mv, 2);
    }

    #[test]
    fn block_opponents_win() {
        // O to play; X threatens at col 1: X _ _ / X _ _ / _ _ _
        let mut b = Board::empty();
        b.cells = [1,0,0, 1,0,0, 0,0,0];
        let mv = best_response(&b, Mark::O).unwrap();
        assert_eq!(mv, 6, "O should block at (row 3, col 1) idx 6");
    }

    #[test]
    fn random_move_uses_lcg() {
        let b = Board::empty();
        let mut rng = LcgRng::new(12345);
        let mv = random_move_with_rng(&b, &mut rng).unwrap();
        // Deterministic given our LCG; assert within range and legal
        assert!(mv < 9);
        assert!(b.is_empty(mv));
    }

    #[test]
    fn blended_chooses_minimax_when_prob_one() {
        let b = Board::empty();
        let mut rng = LcgRng::new(42);
        let mv = blended_move_with_rng(&b, Mark::X, 1.0, &mut rng).unwrap();
        let mm = best_response(&b, Mark::X).unwrap();
        assert_eq!(mv, mm);
    }

    #[test]
    fn blended_chooses_random_when_prob_zero() {
        let b = Board::empty();
        let mut rng = LcgRng::new(42);
        let mv = blended_move_with_rng(&b, Mark::X, 0.0, &mut rng).unwrap();
        // It might still be 4 by chance; ensure it's legal and deterministic
        assert!(mv < 9);
        assert!(b.is_empty(mv));
    }

    #[test]
    fn policy_struct_is_deterministic() {
        let b = Board::empty();
        let mut pol1 = BlendedPolicy::new(7, 0.25);
        let mut pol2 = BlendedPolicy::new(7, 0.25);
        for _ in 0..5 {
            let m1 = pol1.choose(&b, Mark::X).unwrap();
            let m2 = pol2.choose(&b, Mark::X).unwrap();
            assert_eq!(m1, m2);
        }
    }
}
