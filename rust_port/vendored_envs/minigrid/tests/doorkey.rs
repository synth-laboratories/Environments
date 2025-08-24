use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::{Action, Direction};

#[test]
fn doorkey_unlock_and_reach_goal() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Start: (1,1), facing right; key at (2,1); door at (3,1) locked; goal at (3,3)
    let s0 = env.public_state();
    assert_eq!(s0.agent_pos, (1, 1));
    assert_eq!(s0.agent_dir, Direction::Right as u8);

    // Pickup key in front
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_some());

    // Move next to door
    let s2 = env.step(Action::Forward);
    assert_eq!(s2.agent_pos, (2, 1));

    // Toggle door (unlock+open with key)
    let _s3 = env.step(Action::Toggle);

    // Move through door
    let s4 = env.step(Action::Forward);
    assert_eq!(s4.agent_pos, (3, 1));

    // Turn right to face down
    let s5 = env.step(Action::Right);
    assert_eq!(s5.agent_dir, Direction::Down as u8);

    // Step to goal
    let _ = env.step(Action::Forward); // (3,2)
    let sgoal = env.step(Action::Forward); // (3,3)
    assert!(sgoal.terminated, "should terminate on reaching goal");
}

