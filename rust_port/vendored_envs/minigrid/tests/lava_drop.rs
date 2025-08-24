use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::Action;

#[test]
fn stepping_on_lava_terminates() {
    let mut env = MiniGridEnv::lava_inline();
    let s0 = env.public_state();
    assert!(!s0.terminated);
    // Move into lava at (2,1) -> terminate immediately
    let s1 = env.step(Action::Forward);
    assert!(s1.terminated, "should terminate on lava");
}

#[test]
fn drop_only_on_empty_cell() {
    let mut env = MiniGridEnv::doorkey_inline();
    // Pickup key from front
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_some());
    // Try to drop back onto same front cell (now empty): should succeed
    let s2 = env.step(Action::Drop);
    assert!(s2.carrying.is_none());

    // Move forward into the dropped key cell (should be blocked by key)
    let s3 = env.step(Action::Forward);
    assert_eq!(s3.agent_pos, (1,1), "cannot step onto key; must pickup again");
}
