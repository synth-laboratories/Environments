use async_trait::async_trait;
use horizons_core::{register_environment_with_config, EngineError, Environment, Observation, Snapshot, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use sokoban_rs::{Action, Direction, GameConfig, GameState, Level, SimpleLevel, preset_level};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub max_steps: Option<u32>,
    /// When true, use an internal simple static level; otherwise empty level (not yet supported)
    pub use_simple_level: Option<bool>,
    /// Seed for deterministic level generation when not using the fixed simple level
    pub seed: Option<u64>,
    /// Number of boxes to place when seeding a level
    pub num_boxes: Option<usize>,
    /// Optional explicit grids (2D) to construct a level exactly
    pub room_fixed: Option<Vec<Vec<u8>>>,
    pub room_state: Option<Vec<Vec<u8>>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            max_steps: Some(120),
            use_simple_level: Some(true),
            seed: Some(42),
            num_boxes: Some(1),
            room_fixed: None,
            room_state: None,
        }
    }
}

fn map_action_int(a: i64) -> Result<Action, EngineError> {
    // Mirror gym_sokoban's ACTION_LOOKUP exactly:
    // 0: no-op
    // 1..4: push up,down,left,right
    // 5..8: move up,down,left,right
    match a {
        0 => Ok(Action::Noop),
        1 => Ok(Action::Push(Direction::Up)),
        2 => Ok(Action::Push(Direction::Down)),
        3 => Ok(Action::Push(Direction::Left)),
        4 => Ok(Action::Push(Direction::Right)),
        5 => Ok(Action::Move(Direction::Up)),
        6 => Ok(Action::Move(Direction::Down)),
        7 => Ok(Action::Move(Direction::Left)),
        8 => Ok(Action::Move(Direction::Right)),
        _ => Err(EngineError::Validation(format!("unknown action int: {a}"))),
    }
}

pub struct SokobanEnvironment {
    state: GameState,
}

impl SokobanEnvironment {
    pub fn new(config: Config) -> Result<Self, EngineError> {
        let max_steps = config.max_steps.unwrap_or(120);
        // Prefer explicit grids if provided
        let level: Level = if let (Some(fixed2d), Some(state2d)) = (config.room_fixed.clone(), config.room_state.clone()) {
            let h = fixed2d.len();
            if h == 0 { return Err(EngineError::Validation("empty room_fixed".into())); }
            let w = fixed2d[0].len();
            if w == 0 { return Err(EngineError::Validation("empty room_fixed row".into())); }
            if state2d.len() != h || state2d[0].len() != w { return Err(EngineError::Validation("room_fixed/state shape mismatch".into())); }
            let mut lvl = Level { width: w, height: h, room_fixed: Vec::new(), room_state: Vec::new() };
            let mut flat = Vec::with_capacity(w*h);
            for row in fixed2d { flat.extend(row); }
            lvl.room_fixed = flat;
            let mut flat_s = Vec::with_capacity(w*h);
            for row in state2d { flat_s.extend(row); }
            lvl.room_state = flat_s;
            lvl
        } else {
            let seed = config.seed.unwrap_or(42);
            if let Some(preset) = preset_level(seed) {
                preset
            } else if config.use_simple_level.unwrap_or(false) {
                SimpleLevel::build()
            } else {
                let w = config.width.unwrap_or(7);
                let h = config.height.unwrap_or(7);
                let nb = config.num_boxes.unwrap_or(1);
                Level::from_seed(w, h, nb, seed)
            }
        };
        let st = GameState::from_level(&level, max_steps);
        Ok(Self { state: st })
    }

    fn snapshot_obs(&self, extra: Json) -> Observation {
        let public = json!({
            "room_text": self.state.board.room_text(),
            "player_position": [self.state.player_pos.0, self.state.player_pos.1],
            "num_env_steps": self.state.num_env_steps,
            "max_steps": self.state.max_steps,
            "boxes_on_target": self.state.board.boxes_on_target(),
            "num_boxes": self.state.board.num_boxes(),
            "terminated": false, // will be filled by struct fields below
            "truncated": false,
            "extra": extra,
            // reward fields present to be split by PyO3 adapter
            "reward_last": self.state.reward_last,
            "total_reward": self.state.total_reward,
        });
        Observation { terminated: false, truncated: false, data: public }
    }
}

#[async_trait]
impl Environment for SokobanEnvironment {
    async fn initialize(&mut self) -> Result<Observation, EngineError> {
        Ok(self.snapshot_obs(json!({"event":"initialize"})))
    }

    async fn step(&mut self, tool_calls: Vec<ToolCall>) -> Result<Observation, EngineError> {
        if tool_calls.is_empty() {
            return Err(EngineError::Validation("no tool_calls provided".into()));
        }
        let call = &tool_calls[0];
        match call.tool.as_str() {
            "interact" | "move" => {
                let action = if let Some(a) = call.args.get("action").and_then(|v| v.as_i64()) {
                    map_action_int(a)?
                } else {
                    // fall back to direction/mode form
                    let dir = call.args.get("direction").and_then(|v| v.as_str()).ok_or_else(|| EngineError::Validation("missing direction".into()))?;
                    let mode = call.args.get("mode").and_then(|v| v.as_str()).unwrap_or("push");
                    let d = match dir.to_ascii_lowercase().as_str() {
                        "up" => Direction::Up, "down" => Direction::Down, "left" => Direction::Left, "right" => Direction::Right, _ => return Err(EngineError::Validation("invalid direction".into())),
                    };
                    match mode {
                        "move" => Action::Move(d),
                        "auto" => Action::Auto(d),
                        _ => Action::Push(d),
                    }
                };
                let out = self.state.step(action);
                let mut obs = self.snapshot_obs(json!({ "pushed_box": out.pushed_box }));
                obs.terminated = out.terminated;
                obs.truncated = out.truncated;
                // Also update flags in data for convenience
                if let Some(map) = obs.data.as_object_mut() { map.insert("terminated".into(), Json::Bool(obs.terminated)); map.insert("truncated".into(), Json::Bool(obs.truncated)); }
                Ok(obs)
            }
            _ => Err(EngineError::Validation(format!("unknown tool: {}", call.tool))),
        }
    }

    async fn checkpoint(&self) -> Result<Snapshot, EngineError> {
        // Minimal snapshot of visible fields for now; full serde of GameState if needed later
        let data = json!({
            "width": self.state.board.width,
            "height": self.state.board.height,
            "room_fixed": self.state.board.room_fixed,
            "room_state": self.state.board.room_state,
            "player_position": [self.state.player_pos.0, self.state.player_pos.1],
            "num_env_steps": self.state.num_env_steps,
            "max_steps": self.state.max_steps,
            "reward_last": self.state.reward_last,
            "total_reward": self.state.total_reward,
        });
        Ok(Snapshot { version: 1, engine: "sokoban".into(), data })
    }

    async fn terminate(&mut self) -> Result<Observation, EngineError> {
        // No internal state to flip; surface truncated flag
        let mut obs = self.snapshot_obs(json!({"event":"terminate"}));
        obs.truncated = true;
        if let Some(map) = obs.data.as_object_mut() { map.insert("truncated".into(), Json::Bool(true)); }
        Ok(obs)
    }
}

// Registration helper for registry-based construction
pub fn register_default_env() {
    register_environment_with_config(
        "Sokoban",
        Arc::new(|cfg| {
            let cfg: Config = match cfg {
                Some(v) => serde_json::from_value(v).map_err(|e| EngineError::Validation(format!("bad config: {e}")))?,
                None => Config::default(),
            };
            Ok(Box::new(SokobanEnvironment::new(cfg)?))
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizons_core::ToolCall;

    #[tokio::test]
    async fn can_initialize_and_solve_simple() {
        let mut env = SokobanEnvironment::new(Config::default()).unwrap();
        env.initialize().await.unwrap();
        // Push up to place box on target
        let obs = env
            .step(vec![ToolCall { tool: "interact".into(), args: json!({"action":0}) }])
            .await
            .unwrap();
        assert!(obs.terminated, "should be solved after one push up on simple level");
    }
}
