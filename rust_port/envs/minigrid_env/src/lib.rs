use async_trait::async_trait;
use horizons_core::{register_environment_with_config, EngineError, Environment, Observation, Snapshot, ToolCall};
use minigrid_rs::engine::MiniGridEnv;
use minigrid_rs::types::{Action, Direction, ObjectKind};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Config {
    pub env_name: Option<String>,
    pub max_steps: Option<u32>,
    pub seed: Option<u64>,
}

fn make_env(cfg: &Config) -> Result<MiniGridEnv, EngineError> {
    let mut env = match cfg.env_name.as_deref() {
        Some("MiniGrid-Empty-5x5-v0") | Some("empty_5x5") | None => MiniGridEnv::empty_5x5(),
        Some("MiniGrid-DoorKey-5x5-v0") | Some("doorkey_inline") => MiniGridEnv::doorkey_inline(),
        Some("MiniGrid-FourRooms-v0") | Some("four_rooms") => MiniGridEnv::four_rooms_19x19(),
        Some("MiniGrid-Unlock-v0") | Some("unlock_simple") => MiniGridEnv::unlock_simple(),
        Some("MiniGrid-UnlockPickup-v0") | Some("unlockpickup_simple") => MiniGridEnv::unlockpickup_simple(),
        Some(s) => return Err(EngineError::Validation(format!("unsupported env_name: {s}"))),
    };
    if let Some(ms) = cfg.max_steps { env.max_steps = ms; }
    Ok(env)
}

fn action_from_str(name: &str) -> Result<Action, EngineError> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "left" => Action::Left,
        "right" => Action::Right,
        "forward" => Action::Forward,
        "pickup" => Action::Pickup,
        "drop" => Action::Drop,
        "toggle" => Action::Toggle,
        "done" => Action::Done,
        other => return Err(EngineError::Validation(format!("invalid action '{other}'"))),
    })
}

pub struct MiniGridEnvironment {
    env: MiniGridEnv,
}

impl MiniGridEnvironment {
    pub fn new(cfg: Config) -> Result<Self, EngineError> {
        Ok(Self { env: make_env(&cfg)? })
    }

    fn snapshot_obs(&self, event: &str) -> Observation {
        let pubst = self.env.public_state();
        let carrying = pubst.carrying.map(|(k, c)| match k { ObjectKind::Key => json!({"type":"key","color": c}), _ => json!({"type":"other","color": c}) });
        let public = json!({
            "grid_array": pubst.grid_array,
            "agent_pos": [pubst.agent_pos.0, pubst.agent_pos.1],
            "agent_dir": pubst.agent_dir,
            "carrying": carrying,
            "step_count": pubst.step_count,
            "max_steps": pubst.max_steps,
            "mission": pubst.mission,
            "terminated": pubst.terminated,
            "truncated": pubst.step_count >= pubst.max_steps,
            "reward_last": self.env.reward_last(),
            "total_reward": self.env.total_reward(),
            "event": event,
        });
        let terminated = pubst.terminated;
        let truncated = pubst.step_count >= pubst.max_steps;
        Observation { terminated, truncated, data: public }
    }
}

#[async_trait]
impl Environment for MiniGridEnvironment {
    async fn initialize(&mut self) -> Result<Observation, EngineError> {
        Ok(self.snapshot_obs("initialize"))
    }

    async fn step(&mut self, tool_calls: Vec<ToolCall>) -> Result<Observation, EngineError> {
        if tool_calls.is_empty() { return Err(EngineError::Validation("no tool_calls".into())); }
        let call = &tool_calls[0];
        if call.tool != "interact" { return Err(EngineError::Validation(format!("unknown tool: {}", call.tool))); }
        let args = &call.args;
        if let Some(a) = args.get("action").and_then(|v| v.as_str()) {
            let act = action_from_str(a)?;
            let _ = self.env.step(act);
        } else if let Some(arr) = args.get("actions").and_then(|v| v.as_array()) {
            for v in arr {
                let name = v.as_str().ok_or_else(|| EngineError::Validation("actions entries must be strings".into()))?;
                let act = action_from_str(name)?;
                let _ = self.env.step(act);
                if self.env.public_state().terminated { break; }
            }
        } else {
            return Err(EngineError::Validation("missing 'action' or 'actions'".into()));
        }
        Ok(self.snapshot_obs("step"))
    }

    async fn terminate(&mut self) -> Result<Observation, EngineError> {
        Ok(self.snapshot_obs("terminate"))
    }

    async fn checkpoint(&self) -> Result<Snapshot, EngineError> {
        let pubst = self.env.public_state();
        let data = json!({
            "grid_array": pubst.grid_array,
            "agent_pos": [pubst.agent_pos.0, pubst.agent_pos.1],
            "agent_dir": pubst.agent_dir,
            "step_count": pubst.step_count,
            "max_steps": pubst.max_steps,
            "mission": pubst.mission,
        });
        Ok(Snapshot { version: 1, engine: "minigrid".into(), data })
    }
}

pub fn register_default_env() {
    register_environment_with_config(
        "MiniGrid",
        Arc::new(|cfg| {
            let cfg: Config = match cfg { Some(v) => serde_json::from_value(v).map_err(|e| EngineError::Validation(format!("bad config: {e}")))?, None => Config::default(), };
            Ok(Box::new(MiniGridEnvironment::new(cfg)?))
        })
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_rollout_reaches_goal() {
        let mut env = MiniGridEnvironment::new(Config { env_name: Some("MiniGrid-Empty-5x5-v0".into()), max_steps: Some(100), seed: None }).unwrap();
        let _ = env.initialize().await.unwrap();
        let _ = env.step(vec![ToolCall{ tool: "interact".into(), args: json!({"action":"forward"}) }]).await.unwrap();
        let _ = env.step(vec![ToolCall{ tool: "interact".into(), args: json!({"action":"forward"}) }]).await.unwrap();
        let _ = env.step(vec![ToolCall{ tool: "interact".into(), args: json!({"action":"right"}) }]).await.unwrap();
        let _ = env.step(vec![ToolCall{ tool: "interact".into(), args: json!({"actions":["forward","forward"]}) }]).await.unwrap();
        let obs = env.step(vec![ToolCall{ tool: "interact".into(), args: json!({"action":"forward"}) }]).await.unwrap();
        assert!(obs.terminated);
    }
}

