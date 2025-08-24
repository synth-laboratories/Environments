use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::{Action, Direction};

#[test]
fn unlock_simple_flow() {
    let mut env = MiniGridEnv::unlock_simple();

    // Pickup key in front
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_some());

    // Move to the door and toggle (unlock+open)
    for _ in 0..1 { env.step(Action::Forward); }
    env.step(Action::Forward); // approach door row, blocked by wall, stay before door horizontally
    // Face down towards the door
    env.step(Action::Right); // Right from Right -> Down
    let _ = env.step(Action::Toggle);

    // Move through door and head to goal
    let s = env.step(Action::Forward);
    assert_eq!(s.agent_dir, Direction::Down as u8);
}

