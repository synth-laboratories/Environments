use minigrid_rs::types::{Action, Direction, ObjectKind};
use minigrid_rs::engine::MiniGridEnv;

#[test]
fn action_enum_comprehensive_coverage() {
    // Test all action values
    let actions = [
        Action::Left,
        Action::Right,
        Action::Forward,
        Action::Pickup,
        Action::Drop,
        Action::Toggle,
        Action::Done,
    ];

    for (i, &action) in actions.iter().enumerate() {
        assert_eq!(action as u8, i as u8);
        let converted = Action::try_from(i as u8).unwrap();
        assert_eq!(converted as u8, action as u8);
    }
}

#[test]
fn action_try_from_edge_cases() {
    // Test valid range
    for i in 0u8..=6u8 {
        assert!(Action::try_from(i).is_ok());
    }

    // Test invalid values
    for &i in &[7u8, 255u8, 100u8] {
        assert!(Action::try_from(i).is_err());
    }
}

#[test]
fn direction_enum_comprehensive_coverage() {
    let directions = [
        Direction::Right,
        Direction::Down,
        Direction::Left,
        Direction::Up,
    ];

    for (i, &dir) in directions.iter().enumerate() {
        assert_eq!(dir as u8, i as u8);
    }
}

#[test]
fn direction_rotation_comprehensive() {
    let start = Direction::Right;

    // Test full cycles
    assert_eq!(start.right().right().right().right(), start);
    assert_eq!(start.left().left().left().left(), start);

    // Test individual rotations
    assert_eq!(start.right(), Direction::Down);
    assert_eq!(start.right().right(), Direction::Left);
    assert_eq!(start.right().right().right(), Direction::Up);

    assert_eq!(start.left(), Direction::Up);
    assert_eq!(start.left().left(), Direction::Left);
    assert_eq!(start.left().left().left(), Direction::Down);
}

#[test]
fn direction_delta_vectors() {
    assert_eq!(Direction::Right.delta(), (1, 0));
    assert_eq!(Direction::Down.delta(), (0, 1));
    assert_eq!(Direction::Left.delta(), (-1, 0));
    assert_eq!(Direction::Up.delta(), (0, -1));
}

#[test]
fn object_kind_values_comprehensive() {
    // Test all object kind values match Python Gym
    assert_eq!(ObjectKind::Unseen as u8, 0);
    assert_eq!(ObjectKind::Empty as u8, 1);
    assert_eq!(ObjectKind::Wall as u8, 2);
    assert_eq!(ObjectKind::Door as u8, 4);
    assert_eq!(ObjectKind::Key as u8, 5);
    assert_eq!(ObjectKind::Ball as u8, 6);
    assert_eq!(ObjectKind::Goal as u8, 8);
    assert_eq!(ObjectKind::Lava as u8, 9);
    assert_eq!(ObjectKind::Agent as u8, 10);
}

#[test]
fn object_kind_color_encoding() {
    // Test that colors are properly encoded (this is environment-specific)
    // but we can test the basic structure
    let env = MiniGridEnv::empty_5x5();
    let state = env.public_state();

    // Grid should exist and have proper structure
    assert!(!state.grid_array.is_empty());
    assert!(!state.grid_array[0].is_empty());
    assert_eq!(state.grid_array[0][0].len(), 3); // RGB channels
}

#[test]
fn action_string_representation() {
    // Test that actions have reasonable string representations
    // This is more for debugging but ensures the Debug trait works
    let action_str = format!("{:?}", Action::Forward);
    assert!(!action_str.is_empty());
}

#[test]
fn direction_string_representation() {
    let dir_str = format!("{:?}", Direction::Right);
    assert!(!dir_str.is_empty());
}

#[test]
fn action_clone_behavior() {
    let action = Action::Forward;
    let cloned = action.clone();
    assert_eq!(action, cloned);
    assert_eq!(action as u8, cloned as u8);
}

#[test]
fn direction_clone_behavior() {
    let dir = Direction::Left;
    let cloned = dir.clone();
    assert_eq!(dir, cloned);
}

#[test]
fn action_serialization_stability() {
    // Test that action values are stable across runs
    let action_values: Vec<u8> = (0..7).map(|i| Action::try_from(i).unwrap() as u8).collect();
    assert_eq!(action_values, vec![0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn direction_serialization_stability() {
    // Test that direction values are stable
    let dir_values: Vec<u8> = vec![
        Direction::Right as u8,
        Direction::Down as u8,
        Direction::Left as u8,
        Direction::Up as u8,
    ];
    assert_eq!(dir_values, vec![0, 1, 2, 3]);
}

#[test]
fn action_display_formatting() {
    // Test display formatting if available
    let action_str = format!("{:?}", Action::Pickup);
    assert!(action_str.contains("Pickup") || action_str.contains("pickup"));
}

#[test]
fn direction_display_formatting() {
    let dir_str = format!("{:?}", Direction::Down);
    assert!(dir_str.contains("Down") || dir_str.contains("down"));
}

#[test]
fn action_iterators() {
    // Test that we can iterate over all actions
    let all_actions: Vec<Action> = (0..7u8).filter_map(|i| Action::try_from(i).ok()).collect();
    assert_eq!(all_actions.len(), 7);

    // Test specific action types
    assert!(all_actions.contains(&Action::Left));
    assert!(all_actions.contains(&Action::Done));
}

#[test]
fn direction_iterators() {
    // Test that directions can be used in loops
    let all_dirs = [Direction::Right, Direction::Down, Direction::Left, Direction::Up];
    assert_eq!(all_dirs.len(), 4);

    for dir in &all_dirs {
        let delta = dir.delta();
        assert!(delta.0 >= -1 && delta.0 <= 1);
        assert!(delta.1 >= -1 && delta.1 <= 1);
        assert!((delta.0 == 0) != (delta.1 == 0)); // Exactly one should be non-zero
    }
}

#[test]
fn action_from_primitive_comprehensive() {
    // Test all possible u8 conversions
    for i in 0u8..=255u8 {
        let result = Action::try_from(i);
        if i <= 6 {
            assert!(result.is_ok(), "Action {} should be valid", i);
        } else {
            assert!(result.is_err(), "Action {} should be invalid", i);
        }
    }
}

#[test]
fn action_range_bounds() {
    // Test the exact bounds of valid actions
    assert_eq!(Action::try_from(0u8).unwrap() as u8, Action::Left as u8);
    assert_eq!(Action::try_from(6u8).unwrap() as u8, Action::Done as u8);

    assert!(Action::try_from(7u8).is_err());
}

#[test]
fn direction_range_bounds() {
    // Directions are typically 0-3, but test actual implementation
    for i in 0..4 {
        assert!(i == Direction::Right as u8 ||
                i == Direction::Down as u8 ||
                i == Direction::Left as u8 ||
                i == Direction::Up as u8);
    }
}

#[test]
fn action_copy_semantics() {
    // Test that actions can be copied
    let action1 = Action::Toggle;
    let action2 = action1;
    let action3 = action1;

    assert_eq!(action1, action2);
    assert_eq!(action2, action3);
}

#[test]
fn direction_copy_semantics() {
    let dir1 = Direction::Up;
    let dir2 = dir1;
    let dir3 = dir1;

    assert_eq!(dir1, dir2);
    assert_eq!(dir2, dir3);
}

#[test]
fn action_debug_formatting() {
    // Ensure debug formatting works for all actions
    for i in 0..7u8 {
        let action = Action::try_from(i).unwrap();
        let debug_str = format!("{:?}", action);
        assert!(!debug_str.is_empty());
        assert!(!debug_str.contains("Unknown"));
    }
}

#[test]
fn direction_debug_formatting() {
    let directions = [Direction::Right, Direction::Down, Direction::Left, Direction::Up];
    for dir in &directions {
        let debug_str = format!("{:?}", dir);
        assert!(!debug_str.is_empty());
        assert!(!debug_str.contains("Unknown"));
    }
}

#[test]
fn action_partial_eq_comprehensive() {
    // Test all action combinations
    for i in 0..7u8 {
        for j in 0..7u8 {
            let a1 = Action::try_from(i).unwrap();
            let a2 = Action::try_from(j).unwrap();
            assert_eq!(a1 == a2, i == j);
            assert_eq!(a1 != a2, i != j);
        }
    }
}

#[test]
fn direction_partial_eq_comprehensive() {
    let directions = [Direction::Right, Direction::Down, Direction::Left, Direction::Up];
    for &d1 in &directions {
        for &d2 in &directions {
            assert_eq!(d1 == d2, d1 as u8 == d2 as u8);
            assert_eq!(d1 != d2, d1 as u8 != d2 as u8);
        }
    }
}
