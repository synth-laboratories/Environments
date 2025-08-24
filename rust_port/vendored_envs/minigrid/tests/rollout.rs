use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::{Action, Direction};

#[test]
fn empty_5x5_minimal_rollout_reaches_goal() {
    let mut env = MiniGridEnv::empty_5x5();

    // Initial state assertions
    let s0 = env.public_state();
    assert_eq!(s0.agent_pos, (1, 1));
    assert_eq!(s0.agent_dir, Direction::Right as u8);
    assert_eq!(s0.terminated, false);

    // Sequence to reach (3,3) from (1,1) facing Right:
    // forward -> (2,1), forward -> (3,1), right -> face Down,
    // forward -> (3,2), forward -> (3,3) [goal]
    let s1 = env.step(Action::Forward);
    assert_eq!(s1.agent_pos, (2, 1));
    assert!(!s1.terminated);

    let s2 = env.step(Action::Forward);
    assert_eq!(s2.agent_pos, (3, 1));
    assert!(!s2.terminated);

    let s3 = env.step(Action::Right);
    assert_eq!(s3.agent_dir, Direction::Down as u8);
    assert_eq!(s3.agent_pos, (3, 1));
    assert!(!s3.terminated);

    let s4 = env.step(Action::Forward);
    assert_eq!(s4.agent_pos, (3, 2));
    assert!(!s4.terminated);

    let s5 = env.step(Action::Forward);
    assert_eq!(s5.agent_pos, (3, 3));
    assert!(s5.terminated, "should terminate on reaching goal");
    // Reward accounting (MiniGrid): reward = 1 - 0.9*(steps/max_steps)
    // Here steps = 5, max_steps=100 -> 1 - 0.9*(0.05) = 1 - 0.045 = 0.955
    assert_eq!(s5.step_count, 5);
}

#[test]
fn forward_into_wall_is_blocked() {
    let mut env = MiniGridEnv::empty_5x5();
    // Move to (3,1)
    env.step(Action::Forward);
    env.step(Action::Forward);
    let s = env.public_state();
    assert_eq!(s.agent_pos, (3, 1));
    // Next forward would try to move to (4,1) which is a wall/border
    let s_block = env.step(Action::Forward);
    assert_eq!(s_block.agent_pos, (3, 1), "should remain in place when hitting wall");
    assert!(!s_block.terminated);
}
