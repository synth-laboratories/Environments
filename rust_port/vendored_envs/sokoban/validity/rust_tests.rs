use sokoban_rs::{Board, Tile, Action, Direction, GameState};

fn build_board(fixed: &[&[u8]], state: &[&[u8]]) -> (Board, (usize, usize)) {
    let h = fixed.len();
    let w = fixed[0].len();
    let mut b = Board::new(w, h);
    for y in 0..h {
        for x in 0..w {
            b.set_fixed(x, y, tile_from(fixed[y][x]));
            b.set_state(x, y, tile_from(state[y][x]));
        }
    }
    let mut p = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if b.get_state(x, y) == Tile::Player { p = (x, y); }
        }
    }
    (b, p)
}

fn tile_from(code: u8) -> Tile {
    match code {
        0 => Tile::Wall,
        1 => Tile::Empty,
        2 => Tile::Target,
        3 => Tile::BoxOnTarget,
        4 => Tile::Box,
        5 => Tile::Player,
        _ => Tile::Empty,
    }
}

#[test]
fn push_right_solves_board() {
    let fixed = [
        &[1, 1, 1, 1, 1][..],
        &[1, 1, 1, 1, 1][..],
        &[1, 1, 1, 1, 2][..],
        &[1, 1, 1, 1, 1][..],
        &[1, 1, 1, 1, 1][..],
    ];
    let state = [
        &[1, 1, 1, 1, 1][..],
        &[1, 1, 1, 1, 1][..],
        &[1, 1, 5, 4, 1][..],
        &[1, 1, 1, 1, 1][..],
        &[1, 1, 1, 1, 1][..],
    ];
    let (board, player_pos) = build_board(&fixed, &state);
    let mut gs = GameState::new_from_board(board, player_pos, 50);
    let out = gs.step(Action::Auto(Direction::Right));
    assert!(out.moved && out.pushed_box);
    assert!(out.terminated);
    assert!(!out.truncated);
    assert!(gs.reward_last > 0.9);
}

#[test]
fn blocked_move_into_wall_terminates_on_zero_boxes() {
    // Place player adjacent to the top wall; moving up is blocked.
    let fixed = [
        &[0, 0, 0, 0, 0][..],
        &[0, 1, 1, 1, 0][..],
        &[0, 1, 1, 1, 0][..],
        &[0, 1, 1, 1, 0][..],
        &[0, 0, 0, 0, 0][..],
    ];
    let state = [
        &[0, 0, 0, 0, 0][..],
        &[0, 1, 5, 1, 0][..], // player at y=1, x=2
        &[0, 1, 1, 1, 0][..],
        &[0, 1, 1, 1, 0][..],
        &[0, 0, 0, 0, 0][..],
    ];
    let (board, player_pos) = build_board(&fixed, &state);
    let mut gs = GameState::new_from_board(board, player_pos, 10);
    let out = gs.step(Action::Move(Direction::Up));
    assert!(!out.moved);
    assert!(out.terminated);
    assert!(gs.reward_last > 0.9);
}

#[test]
fn push_left_blocked_by_wall_is_noop_and_not_terminated() {
    // Player to the right of a box; wall immediately beyond the box blocks push.
    // Layout:
    // wall | box | player | wall
    let fixed = [
        &[0, 1, 1, 0][..],
        &[0, 1, 1, 0][..],
        &[0, 1, 1, 0][..],
        &[0, 0, 0, 0][..],
    ];
    let state = [
        &[0, 1, 1, 0][..],
        &[0, 4, 5, 0][..], // box at x=1, player at x=2 -> push left blocked by wall at x=0
        &[0, 1, 1, 0][..],
        &[0, 0, 0, 0][..],
    ];
    let (board, player_pos) = build_board(&fixed, &state);
    let mut gs = GameState::new_from_board(board, player_pos, 10);
    let out = gs.step(Action::Auto(Direction::Left));
    assert!(!out.moved && !out.pushed_box);
    assert!(!out.terminated);
    assert!(!out.truncated);
    assert!(gs.reward_last <= -0.01);
}

#[test]
fn truncates_when_exceeding_max_steps_without_solution() {
    let fixed = [
        &[1, 1, 1][..],
        &[1, 1, 1][..],
    ];
    let state = [
        &[1, 4, 1][..],
        &[1, 5, 1][..],
    ];
    let (board, player_pos) = build_board(&fixed, &state);
    let mut gs = GameState::new_from_board(board, player_pos, 1);
    let out = gs.step(Action::Noop);
    assert!(!out.terminated);
    assert!(out.truncated);
    assert!(gs.reward_last <= -0.01);
}
