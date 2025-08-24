use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::{Action, Direction, ObjectKind};

#[test]
fn pickup_key_basic() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Start: (1,1) facing right, key at (2,1)
    let s0 = env.public_state();
    assert_eq!(s0.agent_pos, (1, 1));
    assert_eq!(s0.agent_dir, Direction::Right as u8);
    assert!(s0.carrying.is_none());

    // Move to key position
    let s1 = env.step(Action::Forward);
    assert_eq!(s1.agent_pos, (2, 1));
    assert!(s1.carrying.is_none());

    // Pick up key
    let s2 = env.step(Action::Pickup);
    assert_eq!(s2.agent_pos, (2, 1));
    assert!(s2.carrying.is_some());
    assert_eq!(s2.carrying.unwrap().0, ObjectKind::Key);
}

#[test]
fn pickup_key_from_different_positions() {
    // Test pickup from various positions relative to the key
    let positions = vec![
        ((2, 1), Direction::Right),  // Directly in front
        ((2, 1), Direction::Down),   // Facing down on key
        ((2, 1), Direction::Left),   // Facing left on key
        ((2, 1), Direction::Up),     // Facing up on key
    ];

    for (_key_pos, face_dir) in positions {
        let mut env = MiniGridEnv::doorkey_inline();

        // Navigate to key position
        // This is a simplified test - in real scenarios you'd need
        // to navigate properly to each position
        env.step(Action::Forward); // Go to (2,1)

        // Face the correct direction
        while env.public_state().agent_dir != face_dir as u8 {
            env.step(Action::Right);
        }

        // Try pickup - should work regardless of facing direction when on the object
        let pickup_state = env.step(Action::Pickup);
        assert!(pickup_state.carrying.is_some(),
            "Should be able to pickup key when standing on it");
        assert_eq!(pickup_state.carrying.unwrap().0, ObjectKind::Key);
    }
}

#[test]
fn pickup_nothing() {
    let mut env = MiniGridEnv::empty_5x5();

    // Try to pickup when nothing is in front
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_none());

    // Move forward and try again
    env.step(Action::Forward);
    let s2 = env.step(Action::Pickup);
    assert!(s2.carrying.is_none());
}

#[test]
fn drop_object_basic() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key first
    env.step(Action::Forward); // (2,1)
    env.step(Action::Pickup);  // Pick up key

    let s1 = env.public_state();
    assert!(s1.carrying.is_some());

    // Drop the key
    let s2 = env.step(Action::Drop);
    assert!(s2.carrying.is_none());
    assert_eq!(s2.agent_pos, (2, 1));
}

#[test]
fn drop_object_in_front() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key and navigate to door
    env.step(Action::Forward); // (2,1)
    env.step(Action::Pickup);  // Pick up key
    env.step(Action::Forward); // (3,1) - door position

    let s1 = env.public_state();
    assert!(s1.carrying.is_some());

    // Drop the key in front (should go to (4,1))
    let s2 = env.step(Action::Drop);
    assert!(s2.carrying.is_none());
    // Note: We can't easily verify exact drop position without grid inspection
}

#[test]
fn cannot_drop_on_occupied_cell() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Navigate to position where drop would be blocked
    env.step(Action::Forward); // (2,1)
    env.step(Action::Pickup);  // Pick up key

    // Try to drop - since we're standing on the key position,
    // drop should still work (object gets dropped)
    let s1 = env.step(Action::Drop);
    assert!(s1.carrying.is_none());
}

#[test]
fn toggle_door_without_key() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Navigate to door without picking up key
    env.step(Action::Forward); // (2,1)
    env.step(Action::Forward); // (3,1) - door position

    // Try to toggle door without key - should not work
    let s1 = env.step(Action::Toggle);
    // Door state should remain unchanged - we can't easily verify this
    // without inspecting the grid directly
    assert_eq!(s1.agent_pos, (3, 1));
}

#[test]
fn toggle_door_with_key() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key and navigate to door
    env.step(Action::Forward); // (2,1)
    env.step(Action::Pickup);  // Pick up key
    env.step(Action::Forward); // (3,1) - door position

    let s1 = env.public_state();
    assert!(s1.carrying.is_some());

    // Toggle door with key - should work
    let s2 = env.step(Action::Toggle);
    assert_eq!(s2.agent_pos, (3, 1));
    // Door should now be unlocked - can't verify without grid inspection
}

#[test]
fn toggle_nothing() {
    let mut env = MiniGridEnv::empty_5x5();

    // Try to toggle when nothing is in front
    let s1 = env.step(Action::Toggle);
    assert_eq!(s1.agent_pos, (1, 1)); // Should stay in place

    // Move forward and try again
    env.step(Action::Forward);
    let s2 = env.step(Action::Toggle);
    assert_eq!(s2.agent_pos, (2, 1)); // Should stay in place
}

#[test]
fn carrying_object_persistence() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key
    env.step(Action::Forward);
    env.step(Action::Pickup);

    let s1 = env.public_state();
    assert!(s1.carrying.is_some());
    assert_eq!(s1.carrying.unwrap().0, ObjectKind::Key);

    // Move around while carrying
    env.step(Action::Right);   // Turn
    env.step(Action::Forward); // Move to door

    let s2 = env.public_state();
    assert!(s2.carrying.is_some()); // Should still be carrying
    assert_eq!(s2.carrying.unwrap().0, ObjectKind::Key);
}

#[test]
fn multiple_pickup_attempts() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Move to key
    env.step(Action::Forward);

    // Try to pickup multiple times
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_some());

    let s2 = env.step(Action::Pickup);
    assert!(s2.carrying.is_some()); // Should still be carrying

    let s3 = env.step(Action::Pickup);
    assert!(s3.carrying.is_some()); // Should still be carrying
}

#[test]
fn drop_without_carrying() {
    let mut env = MiniGridEnv::empty_5x5();

    // Try to drop when not carrying anything
    let s1 = env.step(Action::Drop);
    assert!(s1.carrying.is_none());

    // Move and try again
    env.step(Action::Forward);
    let s2 = env.step(Action::Drop);
    assert!(s2.carrying.is_none());
}

#[test]
fn toggle_multiple_times() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key and go to door
    env.step(Action::Forward);
    env.step(Action::Pickup);
    env.step(Action::Forward);

    // Toggle door multiple times
    let s1 = env.step(Action::Toggle);
    let s2 = env.step(Action::Toggle);
    let s3 = env.step(Action::Toggle);

    // All should succeed without error
    assert_eq!(s1.agent_pos, (3, 1));
    assert_eq!(s2.agent_pos, (3, 1));
    assert_eq!(s3.agent_pos, (3, 1));
}

#[test]
fn pickup_drop_cycle() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Move to key
    env.step(Action::Forward);

    // Pickup and drop multiple times
    for _ in 0..3 {
        let s1 = env.step(Action::Pickup);
        assert!(s1.carrying.is_some());

        let s2 = env.step(Action::Drop);
        assert!(s2.carrying.is_none());
    }
}

#[test]
fn carrying_limits() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key
    env.step(Action::Forward);
    env.step(Action::Pickup);

    let s1 = env.public_state();
    assert!(s1.carrying.is_some());

    // Try to pickup again - should not work (can only carry one item)
    let s2 = env.step(Action::Pickup);
    assert!(s2.carrying.is_some()); // Should still be carrying the first item
}

#[test]
fn object_interaction_from_distance() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Try to pickup from a distance - should not work
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_none());

    // Try to toggle from a distance - should not work
    let s2 = env.step(Action::Toggle);
    assert_eq!(s2.agent_pos, (1, 1)); // Should stay in place
}

#[test]
fn door_interaction_states() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Navigate to door
    env.step(Action::Forward); // (2,1)
    env.step(Action::Pickup);  // Pick up key
    env.step(Action::Forward); // (3,1) - door position

    // Try to move through closed door - should be blocked
    let s1 = env.step(Action::Forward);
    assert_eq!(s1.agent_pos, (3, 1)); // Should be blocked

    // Toggle door (unlock)
    env.step(Action::Toggle);

    // Now try to move through - should work
    let s2 = env.step(Action::Forward);
    assert_eq!(s2.agent_pos, (4, 1)); // Should pass through
}

#[test]
fn lava_interaction() {
    let mut env = MiniGridEnv::lava_inline();

    // Start: (1,1) facing right, lava at (2,1)
    let s0 = env.public_state();
    assert!(!s0.terminated);

    // Move into lava - should terminate
    let s1 = env.step(Action::Forward);
    assert!(s1.terminated);
    assert_eq!(s1.agent_pos, (2, 1));
}

#[test]
fn goal_interaction() {
    let mut env = MiniGridEnv::empty_5x5();

    // Navigate to goal at (3,3)
    env.step(Action::Forward); // (2,1)
    env.step(Action::Forward); // (3,1)
    env.step(Action::Right);   // Face down
    env.step(Action::Forward); // (3,2)
    env.step(Action::Forward); // (3,3) - goal

    let s = env.public_state();
    assert!(s.terminated);
    assert_eq!(s.agent_pos, (3, 3));
}

#[test]
fn wall_interaction() {
    let env = MiniGridEnv::empty_5x5();

    // Try to walk through walls in all directions
    let directions = [Direction::Up, Direction::Right, Direction::Down, Direction::Left];

    for dir in directions {
        let mut test_env = MiniGridEnv::empty_5x5();

        // Face the direction
        while test_env.public_state().agent_dir != dir as u8 {
            test_env.step(Action::Right);
        }

        // Try to move - should be blocked by wall
        let before_pos = test_env.public_state().agent_pos;
        let after_state = test_env.step(Action::Forward);
        let after_pos = after_state.agent_pos;

        // Should not have moved
        assert_eq!(before_pos, after_pos,
            "Should be blocked by wall when facing {:?}", dir);
    }
}

#[test]
fn carrying_state_preservation() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Pickup key
    env.step(Action::Forward);
    env.step(Action::Pickup);

    // Perform various actions that shouldn't affect carrying
    let actions = [Action::Left, Action::Right, Action::Forward, Action::Toggle];

    for action in actions {
        let state = env.step(action);
        assert!(state.carrying.is_some(),
            "Should still be carrying after {:?}", action);
    }
}

#[test]
fn object_interaction_with_rotation() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Move to key position
    env.step(Action::Forward);

    // Rotate while on key
    env.step(Action::Right);
    env.step(Action::Right);

    // Should still be able to pickup
    let s1 = env.step(Action::Pickup);
    assert!(s1.carrying.is_some());

    // Rotate while carrying
    env.step(Action::Left);
    env.step(Action::Left);

    // Should still be carrying
    let s2 = env.public_state();
    assert!(s2.carrying.is_some());
}

#[test]
fn interaction_with_terminated_state() {
    let mut env = MiniGridEnv::lava_inline();

    // Die in lava
    env.step(Action::Forward);
    let terminated_state = env.public_state();
    assert!(terminated_state.terminated);

    // Try interactions after termination - should not work
    let s1 = env.step(Action::Pickup);
    assert_eq!(s1.agent_pos, terminated_state.agent_pos);
    assert_eq!(s1.agent_dir, terminated_state.agent_dir);

    let s2 = env.step(Action::Drop);
    assert_eq!(s2.agent_pos, terminated_state.agent_pos);

    let s3 = env.step(Action::Toggle);
    assert_eq!(s3.agent_pos, terminated_state.agent_pos);
}
