use tictactoe_rs::{Board, Mark, apply_move, best_response};

fn play_perfect_game(mut board: Board, mut to_move: Mark) -> Board {
    loop {
        if board.winner().is_some() || board.is_full() { break; }
        if let Some(idx) = best_response(&board, to_move) {
            let ok = apply_move(&mut board, idx, to_move);
            assert!(ok, "minimax chose illegal move at idx {}", idx);
        } else {
            break;
        }
        to_move = to_move.other();
    }
    board
}

#[test]
fn perfect_play_from_start_is_draw() {
    let b = Board::empty();
    let end = play_perfect_game(b, Mark::X);
    assert!(end.winner().is_none(), "perfect play should draw");
    assert!(end.is_full(), "board should be full at terminal draw");
}

#[test]
fn must_block_vertical_threat() {
    // X threatens column 1: indices [0,3] occupied, O to move must play 6
    let mut b = Board::empty();
    b.cells = [1,0,0, 1,0,0, 0,0,0];
    let mv = best_response(&b, Mark::O).unwrap();
    assert_eq!(mv, 6, "O must block at idx 6");
}

#[test]
fn prefer_winning_move_over_block() {
    // X can win on top row; ensure win is taken
    let mut b = Board::empty();
    b.cells = [1,1,0, 0,2,0, 0,0,2];
    let mv = best_response(&b, Mark::X).unwrap();
    assert_eq!(mv, 2, "X should complete the row to win");
}

