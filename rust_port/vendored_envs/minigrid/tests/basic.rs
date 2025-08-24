use minigrid_rs::types::{Action, Direction};

#[test]
fn action_indices_roundtrip() {
    for i in 0u8..=6u8 {
        let a = Action::try_from(i).expect("valid action index");
        assert_eq!(a as u8, i);
    }
    assert!(Action::try_from(7u8).is_err());
}

#[test]
fn direction_rotation_invariants() {
    let start = Direction::Right;
    let d = start.right().right().right().right();
    assert_eq!(d as u8, start as u8);

    let d2 = start.left().left().left().left();
    assert_eq!(d2 as u8, start as u8);
}

