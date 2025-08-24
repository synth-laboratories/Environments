use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::{Action, Direction};

#[test]
fn basic_forward_movement() {
    let env = MiniGridEnv::empty_5x5();
    let s0 = env.public_state();
    assert_eq!(s0.agent_pos, (1, 1));
    assert_eq!(s0.agent_dir, Direction::Right as u8);

    let s1 = env.step(Action::Forward);
    assert_eq!(s1.agent_pos, (2, 1));
    assert_eq!(s1.agent_dir, Direction::Right as u8);
}

#[test]
fn basic_turning_movement() {
    let env = MiniGridEnv::empty_5x5();

    // Turn right
    let s1 = env.step(Action::Right);
    assert_eq!(s1.agent_pos, (1, 1));
    assert_eq!(s1.agent_dir, Direction::Down as u8);

    // Turn right again
    let s2 = env.step(Action::Right);
    assert_eq!(s2.agent_pos, (1, 1));
    assert_eq!(s2.agent_dir, Direction::Left as u8);

    // Turn left
    let s3 = env.step(Action::Left);
    assert_eq!(s3.agent_pos, (1, 1));
    assert_eq!(s3.agent_dir, Direction::Down as u8);
}

#[test]
fn wall_collision_detection() {
    let env = MiniGridEnv::empty_5x5();

    // Move to right edge
    env.step(Action::Forward); // (2,1)
    env.step(Action::Forward); // (3,1)

    // Try to move into wall - should stay in place
    let s = env.step(Action::Forward);
    assert_eq!(s.agent_pos, (3, 1)); // Should not move
}

#[test]
fn boundary_wall_collision() {
    let env = MiniGridEnv::empty_5x5();

    // Face up and try to move into top wall
    env.step(Action::Left); // Face up
    let s = env.step(Action::Forward);
    assert_eq!(s.agent_pos, (1, 1)); // Should not move
}

#[test]
fn corner_navigation() {
    let env = MiniGridEnv::empty_5x5();

    // Navigate to top-right corner: (3,1) facing up
    env.step(Action::Forward); // (2,1)
    env.step(Action::Forward); // (3,1)
    env.step(Action::Left);    // Face up

    // Try all directions from corner - should be blocked on up and right
    let s1 = env.step(Action::Forward); // Try up - blocked
    assert_eq!(s1.agent_pos, (3, 1));

    env.step(Action::Right); // Face right
    let s2 = env.step(Action::Forward); // Try right - blocked
    assert_eq!(s2.agent_pos, (3, 1));

    env.step(Action::Right); // Face down
    let s3 = env.step(Action::Forward); // Try down - should work
    assert_eq!(s3.agent_pos, (3, 2));
}

#[test]
fn full_grid_navigation() {
    let env = MiniGridEnv::empty_5x5();

    // Navigate around the entire grid without hitting walls
    // Start at (1,1) facing right

    // Go right to (3,1)
    env.step(Action::Forward);
    env.step(Action::Forward);

    // Go down to (3,3)
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    // Go left to (1,3)
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    // Go up to (1,1)
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    let final_state = env.public_state();
    assert_eq!(final_state.agent_pos, (1, 1));
}

#[test]
fn movement_with_different_starting_directions() {
    // Test movement starting from different directions
    let directions = [Direction::Right, Direction::Down, Direction::Left, Direction::Up];
    let expected_moves = [
        ((1, 1), Action::Forward, (2, 1)), // Right -> (2,1)
        ((1, 1), Action::Forward, (1, 2)), // Down -> (1,2)
        ((2, 1), Action::Forward, (1, 1)), // Left -> (1,1)
        ((1, 2), Action::Forward, (1, 1)), // Up -> (1,1)
    ];

    for (i, (_start_pos, action, expected_pos)) in expected_moves.iter().enumerate() {
        let env = MiniGridEnv::empty_5x5();

        // Set up starting position if needed
        if i == 2 {
            env.step(Action::Forward); // Move to (2,1) for left test
        } else if i == 3 {
            env.step(Action::Right); // Face down
            env.step(Action::Forward); // Move to (1,2) for up test
        }

        // Set direction
        let target_dir = directions[i];
        let _current_dir = env.public_state().agent_dir;
        let target_dir_u8 = target_dir as u8;

        // Rotate to target direction
        while env.public_state().agent_dir != target_dir_u8 {
            env.step(Action::Right);
        }

        // Perform movement
        let result_state = env.step(*action);
        assert_eq!(result_state.agent_pos, *expected_pos,
            "Movement from direction {:?} should go to {:?}", target_dir, expected_pos);
    }
}

#[test]
fn repeated_movement_patterns() {
    let env = MiniGridEnv::empty_5x5();

    // Test repeated forward movements
    let mut expected_pos = (1, 1);
    for _ in 0..2 { // Can only move 2 steps right before hitting wall
        expected_pos = (expected_pos.0 + 1, expected_pos.1);
        let state = env.step(Action::Forward);
        assert_eq!(state.agent_pos, expected_pos);
    }

    // Third forward should be blocked
    let blocked_state = env.step(Action::Forward);
    assert_eq!(blocked_state.agent_pos, expected_pos);
}

#[test]
fn rotation_without_movement() {
    let env = MiniGridEnv::empty_5x5();

    // Test that rotation doesn't change position
    let start_pos = env.public_state().agent_pos;

    for _ in 0..4 {
        let state = env.step(Action::Right);
        assert_eq!(state.agent_pos, start_pos, "Rotation should not change position");
    }
}

#[test]
fn direction_persistence() {
    let env = MiniGridEnv::empty_5x5();

    // Test that direction persists across multiple actions
    env.step(Action::Right); // Face down
    let dir1 = env.public_state().agent_dir;

    env.step(Action::Left);  // Face right
    env.step(Action::Left);  // Face up
    env.step(Action::Left);  // Face left
    let dir4 = env.public_state().agent_dir;

    assert_ne!(dir1, dir4, "Direction should change with rotations");
}

#[test]
fn movement_interaction_with_rotation() {
    let env = MiniGridEnv::empty_5x5();

    // Test move, rotate, move pattern
    env.step(Action::Forward);     // (2,1)
    env.step(Action::Right);       // Face down, still (2,1)
    env.step(Action::Forward);     // (2,2)

    let state = env.public_state();
    assert_eq!(state.agent_pos, (2, 2));
    assert_eq!(state.agent_dir, Direction::Down as u8);
}

#[test]
fn boundary_edge_cases() {
    let env = MiniGridEnv::empty_5x5();

    // Test all four corners
    let corners = [(1,1), (3,1), (1,3), (3,3)];

    for (x, y) in &corners {
        // Reset environment
        let mut test_env = MiniGridEnv::empty_5x5();

        // Navigate to corner
        while test_env.public_state().agent_pos.0 != *x {
            let state = test_env.step(Action::Forward);
            if state.agent_pos == test_env.public_state().agent_pos {
                break; // Hit wall
            }
        }

        // Set up vertical position
        if *y > 1 {
            test_env.step(Action::Right); // Face down
            for _ in 1..*y {
                let state = test_env.step(Action::Forward);
                if state.agent_pos == test_env.public_state().agent_pos {
                    break; // Hit wall
                }
            }
        }

        let corner_state = test_env.public_state();
        assert_eq!(corner_state.agent_pos, (*x, *y),
            "Should be able to reach corner ({}, {})", x, y);
    }
}

#[test]
fn movement_cost_consistency() {
    let env = MiniGridEnv::empty_5x5();

    // Test that movement and rotation work without errors
    let start_pos = env.public_state().agent_pos;

    // Movement should change position
    let move_state = env.step(Action::Forward);
    assert_ne!(move_state.agent_pos, start_pos);

    // Rotation should work
    let rotate_state = env.step(Action::Right);
    assert!(rotate_state.agent_dir != move_state.agent_dir);
}

#[test]
fn pathfinding_basic() {
    let env = MiniGridEnv::empty_5x5();

    // Simple pathfinding: go right 2, down 2, left 2, up 2
    // Should end up back at start but with some rotation

    // Right 2
    env.step(Action::Forward);
    env.step(Action::Forward);

    // Down 2
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    // Left 2
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    // Up 2
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    let final_state = env.public_state();
    assert_eq!(final_state.agent_pos, (1, 1));
    assert_eq!(final_state.agent_dir, Direction::Up as u8);
}

#[test]
fn obstacle_avoidance_simulation() {
    let mut env = MiniGridEnv::doorkey_inline();

    // Navigate around objects in DoorKey environment
    // Start: (1,1) facing right
    // Key at: (2,1)
    // Door at: (3,1) (locked)
    // Goal at: (3,3)

    // Move to key position
    env.step(Action::Forward); // (2,1) - now on key

    // Can't move forward into wall, so turn around
    env.step(Action::Right);   // Face down
    env.step(Action::Right);   // Face left
    env.step(Action::Forward); // Back to (1,1)

    // Navigate around the long way
    env.step(Action::Right);   // Face down
    env.step(Action::Forward); // (1,2)
    env.step(Action::Forward); // (1,3)
    env.step(Action::Right);   // Face left
    env.step(Action::Forward); // (2,3)
    env.step(Action::Forward); // (3,3) - goal!

    let final_state = env.public_state();
    assert!(final_state.terminated);
    assert_eq!(final_state.agent_pos, (3, 3));
}

#[test]
fn step_count_tracking() {
    let env = MiniGridEnv::empty_5x5();

    // Test step count increments
    assert_eq!(env.public_state().step_count, 0);

    for i in 1..=5 {
        env.step(Action::Forward);
        assert_eq!(env.public_state().step_count, i);
    }
}

#[test]
fn max_steps_termination() {
    let env = MiniGridEnv::empty_5x5();

    // Step until max steps reached
    for _ in 0..env.public_state().max_steps {
        let state = env.step(Action::Right); // Just rotate, don't move
        if state.terminated {
            break;
        }
    }

    let final_state = env.public_state();
    assert!(final_state.terminated);
    assert_eq!(final_state.step_count, final_state.max_steps);
}

#[test]
fn terminated_state_behavior() {
    let env = MiniGridEnv::empty_5x5();

    // Reach goal to terminate
    env.step(Action::Forward);
    env.step(Action::Forward);
    env.step(Action::Right);
    env.step(Action::Forward);
    env.step(Action::Forward);

    let terminated_state = env.public_state();
    assert!(terminated_state.terminated);

    // Actions after termination should return same state
    let post_termination = env.step(Action::Forward);
    assert_eq!(post_termination.agent_pos, terminated_state.agent_pos);
    assert_eq!(post_termination.agent_dir, terminated_state.agent_dir);
    assert_eq!(post_termination.terminated, terminated_state.terminated);
}

#[test]
fn movement_state_consistency() {
    let env = MiniGridEnv::empty_5x5();

    // Test that state is consistent across moves
    let states = vec![
        env.step(Action::Forward),
        env.step(Action::Right),
        env.step(Action::Forward),
        env.step(Action::Left),
        env.step(Action::Forward),
    ];

    // Each state should have valid values
    for (i, state) in states.iter().enumerate() {
        assert!(state.agent_pos.0 >= 1 && state.agent_pos.0 <= 3,
            "State {} has invalid x position: {}", i, state.agent_pos.0);
        assert!(state.agent_pos.1 >= 1 && state.agent_pos.1 <= 3,
            "State {} has invalid y position: {}", i, state.agent_pos.1);
        assert!(state.agent_dir <= 3,
            "State {} has invalid direction: {}", i, state.agent_dir);
        assert!(state.step_count >= 0,
            "State {} has negative step count: {}", i, state.step_count);
    }
}

#[test]
fn diagonal_movement_impossible() {
    let env = MiniGridEnv::empty_5x5();

    // Test that diagonal movement is impossible
    // Agent can only move in cardinal directions
    let start_pos = env.public_state().agent_pos;

    // Try to move diagonally by alternating forward and turn
    env.step(Action::Forward);
    env.step(Action::Right);
    env.step(Action::Forward);

    let end_pos = env.public_state().agent_pos;

    // Should be at (2,2), not some diagonal position
    assert_eq!(end_pos, (2, 2));
    assert_ne!(end_pos.0, start_pos.0 + 1, "Should not have moved diagonally");
    assert_ne!(end_pos.1, start_pos.1 + 1, "Should not have moved diagonally");
}

#[test]
fn movement_range_validation() {
    let env = MiniGridEnv::empty_5x5();

    // Test movement within valid range
    // In 5x5 grid, valid positions are (1,1) to (3,3)
    let valid_positions = vec![
        (1,1), (2,1), (3,1),
        (1,2), (2,2), (3,2),
        (1,3), (2,3), (3,3),
    ];

    for (x, y) in valid_positions {
        // Reset and navigate to position
        let mut test_env = MiniGridEnv::empty_5x5();

        // Navigate to x position
        for _ in 1..x {
            test_env.step(Action::Forward);
        }

        if y > 1 {
            test_env.step(Action::Right); // Face down
            for _ in 1..y {
                test_env.step(Action::Forward);
            }
        }

        let state = test_env.public_state();
        assert_eq!(state.agent_pos, (x, y),
            "Should be able to reach valid position ({}, {})", x, y);
    }
}

#[test]
fn invalid_position_impossible() {
    // Test that agent can never reach invalid positions
    let env = MiniGridEnv::empty_5x5();

    // Try many random action sequences
    for _ in 0..100 {
        env.step(Action::Forward);
        env.step(Action::Right);
        env.step(Action::Left);
    }

    let final_pos = env.public_state().agent_pos;

    // Position should always be within valid bounds
    assert!(final_pos.0 >= 1 && final_pos.0 <= 3);
    assert!(final_pos.1 >= 1 && final_pos.1 <= 3);
}

#[test]
fn movement_determinism() {
    // Test that same action sequence produces same results
    let actions = vec![
        Action::Forward, Action::Right, Action::Forward,
        Action::Left, Action::Forward, Action::Right,
        Action::Forward, Action::Left, Action::Forward,
    ];

    let mut env1 = MiniGridEnv::empty_5x5();
    let mut env2 = MiniGridEnv::empty_5x5();

    for action in actions {
        env1.step(action);
        env2.step(action);

        let state1 = env1.public_state();
        let state2 = env2.public_state();

        assert_eq!(state1.agent_pos, state2.agent_pos);
        assert_eq!(state1.agent_dir, state2.agent_dir);
        assert_eq!(state1.step_count, state2.step_count);
    }
}

#[test]
fn state_equality_after_identical_actions() {
    let mut env1 = MiniGridEnv::empty_5x5();
    let mut env2 = MiniGridEnv::empty_5x5();

    // Perform identical action sequences
    let actions = [Action::Forward, Action::Right, Action::Forward, Action::Left];

    for action in &actions {
        env1.step(*action);
        env2.step(*action);
    }

    let state1 = env1.public_state();
    let state2 = env2.public_state();

    assert_eq!(state1.agent_pos, state2.agent_pos);
    assert_eq!(state1.agent_dir, state2.agent_dir);
    assert_eq!(state1.step_count, state2.step_count);
    assert_eq!(state1.terminated, state2.terminated);
}

#[test]
fn movement_speed_consistency() {
    let env = MiniGridEnv::empty_5x5();

    // Test that movement speed is always 1 step
    let start_pos = env.public_state().agent_pos;

    env.step(Action::Forward);
    let after_one = env.public_state().agent_pos;

    // Distance should be exactly 1
    let distance = ((after_one.0 as i32 - start_pos.0 as i32).abs() +
                   (after_one.1 as i32 - start_pos.1 as i32).abs()) as u32;
    assert_eq!(distance, 1, "Movement should always be exactly 1 step");
}

#[test]
fn rotation_does_not_change_position() {
    let env = MiniGridEnv::empty_5x5();

    // Test that any sequence of rotations doesn't change position
    let start_pos = env.public_state().agent_pos;

    let rotations = [Action::Left, Action::Right, Action::Left, Action::Right];
    for rotation in &rotations {
        env.step(*rotation);
        assert_eq!(env.public_state().agent_pos, start_pos,
            "Rotation should never change position");
    }
}

#[test]
fn movement_always_changes_position_when_possible() {
    let env = MiniGridEnv::empty_5x5();

    // In the center, movement should always work
    let center_pos = (2, 2);

    // Navigate to center
    env.step(Action::Forward); // (2,1)
    env.step(Action::Right);   // Face down
    env.step(Action::Forward); // (2,2)

    assert_eq!(env.public_state().agent_pos, center_pos);

    // From center, movement in any direction should work
    let directions = [Direction::Right, Direction::Down, Direction::Left, Direction::Up];
    let expected_positions = [(3, 2), (2, 3), (1, 2), (2, 1)];

    for (i, expected_pos) in expected_positions.iter().enumerate() {
        let mut center_env = MiniGridEnv::empty_5x5();
        center_env.step(Action::Forward);
        center_env.step(Action::Right);
        center_env.step(Action::Forward);

        // Rotate to correct direction
        for _ in 0..i {
            center_env.step(Action::Right);
        }

        let move_state = center_env.step(Action::Forward);
        assert_eq!(move_state.agent_pos, *expected_pos,
            "From center, should be able to move to {:?}", expected_pos);
    }
}

#[test]
fn wall_detection_from_all_directions() {
    let env = MiniGridEnv::empty_5x5();

    // Test wall detection from all four walls
    let wall_tests = vec![
        ((1,1), Direction::Left, true),   // Left wall at (1,1)
        ((3,1), Direction::Right, true),  // Right wall at (3,1)
        ((1,1), Direction::Up, true),     // Top wall at (1,1)
        ((1,3), Direction::Down, true),   // Bottom wall at (1,3)
        ((2,2), Direction::Right, false), // No wall at (2,2) right
        ((2,2), Direction::Down, false),  // No wall at (2,2) down
    ];

    for (start_pos, face_dir, should_be_blocked) in wall_tests {
        let mut test_env = MiniGridEnv::empty_5x5();

        // Navigate to start position
        if start_pos.0 > 1 {
            for _ in 1..start_pos.0 {
                test_env.step(Action::Forward);
            }
        }
        if start_pos.1 > 1 {
            test_env.step(Action::Right);
            for _ in 1..start_pos.1 {
                test_env.step(Action::Forward);
            }
        }

        // Face the correct direction
        let current_dir = test_env.public_state().agent_dir;
        let target_dir = face_dir as u8;

        while current_dir != target_dir {
            test_env.step(Action::Right);
        }

        // Try to move
        let before_pos = test_env.public_state().agent_pos;
        test_env.step(Action::Forward);
        let after_pos = test_env.public_state().agent_pos;

        if should_be_blocked {
            assert_eq!(before_pos, after_pos,
                "Should be blocked by wall at {:?}", start_pos);
        } else {
            assert_ne!(before_pos, after_pos,
                "Should be able to move from {:?}", start_pos);
        }
    }
}

#[test]
fn movement_undo_simulation() {
    let env = MiniGridEnv::empty_5x5();

    // Test "undoing" movement with opposite actions
    let start_state = env.public_state();

    // Move forward, then backward (turn around and move)
    env.step(Action::Forward);     // Move right
    env.step(Action::Right);       // Face down
    env.step(Action::Right);       // Face left
    env.step(Action::Forward);     // Move left (back)

    let final_state = env.public_state();

    // Should be back to start position but facing left
    assert_eq!(final_state.agent_pos, start_state.agent_pos);
    assert_eq!(final_state.agent_dir, Direction::Left as u8);
}

#[test]
fn complex_navigation_pattern() {
    let env = MiniGridEnv::empty_5x5();

    // Perform a complex navigation pattern
    let pattern = vec![
        Action::Forward, Action::Right, Action::Forward, Action::Forward,
        Action::Right, Action::Forward, Action::Left, Action::Forward,
        Action::Right, Action::Forward, Action::Forward, Action::Left,
    ];

    let mut positions = vec![];
    for action in pattern {
        let state = env.step(action);
        positions.push(state.agent_pos);
    }

    // All positions should be valid
    for (i, pos) in positions.iter().enumerate() {
        assert!(pos.0 >= 1 && pos.0 <= 3,
            "Position {} x-coordinate invalid: {}", i, pos.0);
        assert!(pos.1 >= 1 && pos.1 <= 3,
            "Position {} y-coordinate invalid: {}", i, pos.1);
    }
}

#[test]
fn action_sequence_reproducibility() {
    // Test that the same action sequence always produces the same result
    let action_sequence = vec![
        Action::Forward, Action::Right, Action::Forward, Action::Left,
        Action::Forward, Action::Right, Action::Forward, Action::Left,
        Action::Forward, Action::Right, Action::Forward,
    ];

    let mut results = vec![];

    // Run the sequence 5 times
    for _ in 0..5 {
        let env = MiniGridEnv::empty_5x5();
        for action in &action_sequence {
            env.step(*action);
        }
        results.push(env.public_state().agent_pos);
    }

    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i],
            "Sequence should produce identical results: run 0 = {:?}, run {} = {:?}", results[0], i, results[i]);
    }
}

#[test]
fn grid_boundary_awareness() {
    let env = MiniGridEnv::empty_5x5();

    // Test that agent is aware of grid boundaries
    // In a 5x5 grid, the playable area is 3x3 (positions 1,1 to 3,3)

    // Try to go out of bounds in all directions
    let boundary_tests = vec![
        ((1,1), Direction::Left, true),   // Can't go left from (1,1)
        ((3,1), Direction::Right, true),  // Can't go right from (3,1)
        ((1,1), Direction::Up, true),     // Can't go up from (1,1)
        ((1,3), Direction::Down, true),   // Can't go down from (1,3)
    ];

    for (pos, dir, should_block) in boundary_tests {
        let mut test_env = MiniGridEnv::empty_5x5();

        // Navigate to position
        if pos.0 > 1 {
            for _ in 1..pos.0 {
                test_env.step(Action::Forward);
            }
        }
        if pos.1 > 1 {
            test_env.step(Action::Right);
            for _ in 1..pos.1 {
                test_env.step(Action::Forward);
            }
        }

        // Face direction
        while test_env.public_state().agent_dir != dir as u8 {
            test_env.step(Action::Right);
        }

        // Try to move
        let before = test_env.public_state().agent_pos;
        test_env.step(Action::Forward);
        let after = test_env.public_state().agent_pos;

        if should_block {
            assert_eq!(before, after,
                "Should be blocked at boundary position {:?}", pos);
        }
    }
}
