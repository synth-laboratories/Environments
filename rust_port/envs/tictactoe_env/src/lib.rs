use async_trait::async_trait;
use horizons_core::{
    register_environment_with_config, EngineError, Environment, Observation, Snapshot, ToolCall,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use std::sync::Arc;
use tictactoe_rs::{apply_move, best_response, blended_move_with_rng, Board, BlendedPolicy, LcgRng, Mark};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub agent_mark: String,        // "X" or "O"
    pub opponent_minimax_prob: f64, // 0.0 random .. 1.0 minimax
    pub seed: u64,                 // RNG seed for opponent policy
}

impl Default for Config {
    fn default() -> Self {
        Self { agent_mark: "X".into(), opponent_minimax_prob: 0.0, seed: 42 }
    }
}

fn parse_mark(s: &str) -> Result<Mark, EngineError> {
    match s.to_ascii_uppercase().as_str() {
        "X" => Ok(Mark::X),
        "O" => Ok(Mark::O),
        _ => Err(EngineError::Validation(format!("invalid mark: {}", s))),
    }
}

pub struct TicTacToeEnvironment {
    board: Board,
    to_move: Mark,
    agent_mark: Mark,
    opponent: BlendedPolicy,
    terminated: bool,
    truncated: bool,
    move_count: u32,
    last_move: Option<String>,
    reward_last: f64,
    total_reward: f64,
}

impl TicTacToeEnvironment {
    pub fn new(config: Config) -> Result<Self, EngineError> {
        let agent_mark = parse_mark(&config.agent_mark)?;
        let opponent = BlendedPolicy::new(config.seed, config.opponent_minimax_prob);
        Ok(Self {
            board: Board::empty(),
            to_move: Mark::X,
            agent_mark,
            opponent,
            terminated: false,
            truncated: false,
            move_count: 0,
            last_move: None,
            reward_last: 0.0,
            total_reward: 0.0,
        })
    }

    fn legal_moves(&self) -> Vec<usize> {
        let (arr, n) = tictactoe_rs::legal_moves_array(&self.board);
        arr[..n].to_vec()
    }

    fn winner(&self) -> Option<Mark> { self.board.winner() }

    fn snapshot_obs(&self, extra: Json) -> Observation {
        let winner = self
            .winner()
            .map(|m| match m {
                Mark::X => json!("X"),
                Mark::O => json!("O"),
            })
            .unwrap_or(Json::Null);

        let current_player = match self.to_move { Mark::X => "X", Mark::O => "O" };
        Observation {
            terminated: self.terminated,
            truncated: self.truncated,
            data: json!({
                // Spec-aligned public observation fields
                "board_text": self.board.board_text(),
                "current_player": current_player,
                "move_count": self.move_count,
                "last_move": self.last_move,
                "winner": winner,
                // Reward fields
                "reward_last": self.reward_last,
                "total_reward": self.total_reward,
                // Extra section for tool-specific data when needed
                "extra": extra,
            }),
        }
    }

    fn apply_agent_move(&mut self, idx: usize) -> Result<(), EngineError> {
        if self.terminated {
            return Err(EngineError::Validation("game already over".into()));
        }
        if self.to_move != self.agent_mark {
            return Err(EngineError::Validation("not agent's turn".into()));
        }
        if idx >= 9 || !self.board.is_empty(idx) {
            // Spec: illegal move ends episode with negative reward
            self.terminated = true;
            self.reward_last = -1.0;
            self.total_reward += self.reward_last;
            return Ok(());
        }
        apply_move(&mut self.board, idx, self.agent_mark);
        self.last_move = Board::idx_to_coord(idx);
        self.move_count += 1;
        // Win/draw check after agent move
        if let Some(w) = self.board.winner() {
            self.terminated = true;
            // Reward from agent perspective
            self.reward_last = if w == self.agent_mark { 1.0 } else { -1.0 };
            self.total_reward += self.reward_last;
            return Ok(());
        }
        if self.board.is_draw() {
            self.terminated = true;
            self.reward_last = 0.0;
            self.total_reward += self.reward_last;
            return Ok(());
        }
        // Otherwise hand over to opponent
        self.to_move = self.agent_mark.other();
        Ok(())
    }

    fn apply_opponent_move(&mut self) {
        if self.terminated { return; }
        if self.to_move == self.agent_mark { return; }
        if let Some(idx) = self.opponent.choose(&self.board, self.to_move) {
            let _ = apply_move(&mut self.board, idx, self.to_move);
            self.last_move = Board::idx_to_coord(idx);
            self.move_count += 1;
        }
        // Check terminal after opponent move
        if let Some(w) = self.board.winner() {
            self.terminated = true;
            // If opponent wins, agent loses => -1.0
            self.reward_last = if w == self.agent_mark { 1.0 } else { -1.0 };
            self.total_reward += self.reward_last;
        } else if self.board.is_draw() {
            self.terminated = true;
            self.reward_last = 0.0;
            self.total_reward += self.reward_last;
        } else {
            // Otherwise back to agent
            self.to_move = self.to_move.other();
            // No terminal => no reward assigned this full step yet (keep last)
        }
    }
}

#[async_trait]
impl Environment for TicTacToeEnvironment {
    async fn initialize(&mut self) -> Result<Observation, EngineError> { Ok(self.snapshot_obs(json!({"event":"initialize"}))) }

    async fn step(&mut self, tool_calls: Vec<ToolCall>) -> Result<Observation, EngineError> {
        if tool_calls.is_empty() {
            return Err(EngineError::Validation("no tool_calls provided".into()));
        }
        let call = &tool_calls[0];
        match call.tool.as_str() {
            // Spec-aligned primary tool
            "interact" => {
                let letter = call.args.get("letter").and_then(|v| v.as_str()).ok_or_else(|| EngineError::Validation("missing letter".into()))?;
                let number = call.args.get("number").and_then(|v| v.as_i64()).ok_or_else(|| EngineError::Validation("missing number".into()))?;
                let idx = Board::coord_to_idx(letter, number).ok_or_else(|| EngineError::Validation("invalid coordinate".into()))?;
                self.apply_agent_move(idx)?;
                if !self.terminated { self.apply_opponent_move(); }
                if !self.terminated { self.reward_last = 0.0; }
                Ok(self.snapshot_obs(json!({
                    "agent_move_index": idx,
                    "legal_moves": self.legal_moves(),
                })))
            }
            // Backwards-compatible aliases helpful for debugging
            "place" => {
                let idx = call.args.get("index").and_then(|v| v.as_u64()).ok_or_else(|| EngineError::Validation("missing index".into()))? as usize;
                self.apply_agent_move(idx)?;
                if !self.terminated { self.apply_opponent_move(); }
                if !self.terminated { self.reward_last = 0.0; }
                Ok(self.snapshot_obs(json!({
                    "agent_move_index": idx,
                    "legal_moves": self.legal_moves(),
                })))
            }
            "place_coord" => {
                let col = call.args.get("col").and_then(|v| v.as_str()).ok_or_else(|| EngineError::Validation("missing col".into()))?;
                let row = call.args.get("row").and_then(|v| v.as_i64()).ok_or_else(|| EngineError::Validation("missing row".into()))?;
                let idx = Board::coord_to_idx(col, row).ok_or_else(|| EngineError::Validation("invalid coordinate".into()))?;
                self.apply_agent_move(idx)?;
                if !self.terminated { self.apply_opponent_move(); }
                if !self.terminated { self.reward_last = 0.0; }
                Ok(self.snapshot_obs(json!({
                    "agent_move_index": idx,
                    "legal_moves": self.legal_moves(),
                })))
            }
            "suggest" => {
                let mv = best_response(&self.board, self.to_move);
                Ok(self.snapshot_obs(json!({"suggested": mv})))
            }
            _ => Err(EngineError::Validation(format!("unknown tool: {}", call.tool))),
        }
    }

    async fn checkpoint(&self) -> Result<Snapshot, EngineError> {
        let data = json!({
            "cells": self.board.cells,
            "to_move": match self.to_move { Mark::X=>"X", Mark::O=>"O" },
            "agent_mark": match self.agent_mark { Mark::X=>"X", Mark::O=>"O" },
            "move_count": self.move_count,
            "last_move": self.last_move,
            "reward_last": self.reward_last,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        });
        Ok(Snapshot { version: 1, engine: "tictactoe".into(), data })
    }

    async fn terminate(&mut self) -> Result<Observation, EngineError> {
        self.truncated = true;
        Ok(self.snapshot_obs(json!({"event":"terminate"})))
    }
}

// Registration helper so callers can create via core registry
pub fn register_default_env() {
    register_environment_with_config(
        "TicTacToe",
        Arc::new(|cfg| {
            let cfg: Config = match cfg {
                Some(v) => serde_json::from_value(v).map_err(|e| EngineError::Validation(format!("bad config: {e}")))?,
                None => Config::default(),
            };
            Ok(Box::new(TicTacToeEnvironment::new(cfg)?))
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn suggest_matches_minimax() {
        let mut env = TicTacToeEnvironment::new(Config::default()).unwrap();
        env.initialize().await.unwrap();
        // X at corner, expect O would prefer center (suggest for O after switching)
        env.agent_mark = Mark::X;
        env.apply_agent_move(0).unwrap(); // X at A1
        // Now to_move is O; ask suggest
        let obs = env
            .step(vec![ToolCall { tool: "suggest".into(), args: json!({}) }])
            .await
            .unwrap();
        let suggested = obs.data["extra"]["suggested"].as_u64().unwrap() as usize;
        assert_eq!(suggested, 4);
    }

    #[tokio::test]
    async fn random_opponent_differs_by_seed() {
        let mut env_a = TicTacToeEnvironment::new(Config { seed: 1, ..Default::default() }).unwrap();
        let mut env_b = TicTacToeEnvironment::new(Config { seed: 2, ..Default::default() }).unwrap();
        env_a.initialize().await.unwrap();
        env_b.initialize().await.unwrap();
        // Agent X places center in both
        let obs_a = env_a
            .step(vec![ToolCall { tool: "place".into(), args: json!({"index":4}) }])
            .await
            .unwrap();
        let obs_b = env_b
            .step(vec![ToolCall { tool: "place".into(), args: json!({"index":4}) }])
            .await
            .unwrap();
        // After agent move, opponent moved; detect opponent move by diffing legal moves count
        let lm_a: Vec<_> = obs_a.data["legal_moves"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let lm_b: Vec<_> = obs_b.data["legal_moves"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_ne!(lm_a, lm_b, "different seeds should yield different opponent moves with random policy");
    }

    #[tokio::test]
    async fn opponent_blocks_with_minimax() {
        let mut env = TicTacToeEnvironment::new(Config { opponent_minimax_prob: 1.0, ..Default::default() }).unwrap();
        env.initialize().await.unwrap();
        // X opens at A1, then X at A2 to threaten column 1; O should block at A3 (idx 6)
        env.step(vec![ToolCall { tool: "place".into(), args: json!({"index":0}) }])
            .await
            .unwrap();
        // Now it's X again
        let obs = env
            .step(vec![ToolCall { tool: "place".into(), args: json!({"index":3}) }])
            .await
            .unwrap();
        // O should have just moved; ensure cell 6 is not empty anymore
        // The env returns current legal moves; 6 should be absent
        let legal: Vec<_> = obs.data["extra"]["legal_moves"].as_array().unwrap_or(&vec![]).iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert!(!legal.contains(&6), "O should block at idx 6 under minimax");
    }

    #[tokio::test]
    async fn config_init_and_minimax_block() {
        // Register configurable factory and initialize via config JSON
        register_default_env();
        let cfg = json!({"agent_mark":"X","opponent_minimax_prob":1.0,"seed":7});
        let mut env = horizons_core::create_environment_with_config("TicTacToe", Some(cfg)).unwrap();
        env.initialize().await.unwrap();
        // Agent X opens A1
        env.step(vec![ToolCall { tool: "place".into(), args: json!({"index":0}) }])
            .await
            .unwrap();
        // Agent X plays A2; opponent O (minimax) should block at A3 (idx 6)
        let obs = env
            .step(vec![ToolCall { tool: "place".into(), args: json!({"index":3}) }])
            .await
            .unwrap();
        let legal: Vec<_> = obs.data["extra"]["legal_moves"].as_array().unwrap_or(&vec![]).iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert!(!legal.contains(&6), "O should block at idx 6 under minimax when initialized via config");
    }
}
