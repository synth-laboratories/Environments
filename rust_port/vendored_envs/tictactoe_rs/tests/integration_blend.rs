use tictactoe_rs::{
    Board, Mark, best_response, LcgRng,
    blended_move_with_rng, BlendedPolicy, random_move_with_rng,
};

fn find_seed_for_choice(p: f64, player: Mark, board: &Board, want_minimax: bool) -> Option<u64> {
    // Search a modest range to find a seed that drives the first roll below/above p
    for seed in 0u64..5000 {
        let mut rng = LcgRng::new(seed);
        let mv = blended_move_with_rng(board, player, p, &mut rng);
        if mv.is_none() { continue; }
        let mm = best_response(board, player).unwrap();
        let is_mm = mv.unwrap() == mm;
        if is_mm == want_minimax {
            return Some(seed);
        }
    }
    None
}

#[test]
fn blended_prob_one_matches_minimax() {
    let b = Board::empty();
    let mut rng = LcgRng::new(123);
    let mv = blended_move_with_rng(&b, Mark::X, 1.0, &mut rng).unwrap();
    let mm = best_response(&b, Mark::X).unwrap();
    assert_eq!(mv, mm);
}

#[test]
fn blended_prob_zero_is_random_legal() {
    let b = Board::empty();
    let mut rng = LcgRng::new(123);
    let mv = blended_move_with_rng(&b, Mark::X, 0.0, &mut rng).unwrap();
    assert!(mv < 9 && b.is_empty(mv));
}

#[test]
fn can_force_random_or_minimax_via_seed() {
    let b = Board::empty();
    let p = 0.5;
    let mm_seed = find_seed_for_choice(p, Mark::X, &b, true).expect("seed for mm branch");
    let rnd_seed = find_seed_for_choice(p, Mark::X, &b, false).expect("seed for random branch");
    // Sanity: both seeds found and distinct
    assert_ne!(mm_seed, rnd_seed);
}

#[test]
fn policy_sequence_is_deterministic() {
    let mut pol1 = BlendedPolicy::new(42, 0.3);
    let mut pol2 = BlendedPolicy::new(42, 0.3);
    let mut b1 = Board::empty();
    let mut b2 = Board::empty();
    let mut to_move = Mark::X;
    for _ in 0..5 {
        let m1 = pol1.choose(&b1, to_move).unwrap();
        let m2 = pol2.choose(&b2, to_move).unwrap();
        assert_eq!(m1, m2);
        // Apply to both boards
        tictactoe_rs::apply_move(&mut b1, m1, to_move);
        tictactoe_rs::apply_move(&mut b2, m2, to_move);
        to_move = to_move.other();
    }
}

#[test]
fn random_move_respects_legality() {
    // Fill a few cells and ensure random never picks them.
    let mut b = Board::empty();
    b.cells = [1,0,0, 2,0,0, 0,0,0];
    let mut rng = LcgRng::new(999);
    for _ in 0..9 {
        match random_move_with_rng(&b, &mut rng) {
            Some(mv) => {
                assert!(b.is_empty(mv));
                tictactoe_rs::apply_move(&mut b, mv, Mark::X);
            }
            None => break,
        }
    }
}
