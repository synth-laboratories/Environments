use sokoban_rs::{GameState, GameConfig, Action, Direction, SimpleLevel};

#[test]
fn simple_level_push_to_target() {
    let lvl = SimpleLevel::build();
    let mut gs = GameState::from_level(&lvl, GameConfig::default().max_steps);
    // Initial: player at (2,3), box at (2,2), target at (2,1).
    assert_eq!(gs.board.num_boxes(), 1);
    assert_eq!(gs.board.boxes_on_target(), 0);

    // Push up twice: first push moves box to (2,1) which is a target
    let out1 = gs.step(Action::Push(Direction::Up));
    assert!(out1.moved && out1.pushed_box);
    assert_eq!(gs.board.boxes_on_target(), 1);
    assert!(out1.terminated);

    // Now the box is on target; pushing again should fail (wall behind)
    let out2 = gs.step(Action::Push(Direction::Up));
    assert!(!out2.pushed_box);

    // Solved condition equals boxes_on_target == num_boxes
    let solved = gs.board.boxes_on_target() == gs.board.num_boxes();
    assert!(solved);
}

#[test]
fn room_text_renders() {
    let lvl = SimpleLevel::build();
    let gs = GameState::from_level(&lvl, GameConfig::default().max_steps);
    let s = gs.board.room_text();
    assert!(s.contains('#'));
    assert!(s.contains('P'));
}
