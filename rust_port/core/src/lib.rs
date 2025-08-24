//! Core traits and types for Horizons Rust port.
//! Contracts mirror rust_port/rewrite.txt (ToolCall, Observation, Snapshot, Environment).

use async_trait::async_trait;
use std::sync::OnceLock;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Canonical tool call: tool name and JSON-serializable arguments.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub tool: String,
    #[serde(default)]
    pub args: Json,
}

/// Result of a tool invocation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub ok: bool,
    #[serde(default)]
    pub payload: Option<Json>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Observation contract. Enforces presence of terminated/truncated; additional fields live in `data`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    pub terminated: bool,
    pub truncated: bool,
    /// Per-environment fields (e.g., board_text, room_text, reward_last, etc.).
    #[serde(default)]
    pub data: Json,
}

/// Snapshot contract for checkpoint/restore.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Snapshot {
    pub version: u32,
    pub engine: String,
    pub data: Json,
}

/// Environment errors mapped to HTTP responses by services.
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
}

/// Core async environment trait.
#[async_trait]
pub trait Environment: Send + Sync {
    async fn initialize(&mut self) -> Result<Observation, EngineError>;
    async fn step(&mut self, tool_calls: Vec<ToolCall>) -> Result<Observation, EngineError>;
    async fn checkpoint(&self) -> Result<Snapshot, EngineError>;
    async fn terminate(&mut self) -> Result<Observation, EngineError>;
}

// --------------------------
// Tools: trait + registry
// --------------------------

/// Trait for environment tools callable by agents.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    async fn call(&self, call: &ToolCall) -> Result<ToolResult, EngineError>;
}

static TOOL_REGISTRY: OnceLock<Mutex<HashMap<String, Arc<dyn Tool>>>> = OnceLock::new();

/// Register a tool instance by name. Overwrites any existing entry.
pub fn register_tool(tool: Arc<dyn Tool>) {
    let mut reg = TOOL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("tool registry poisoned");
    reg.insert(tool.name().to_string(), tool);
}

/// Fetch a tool by name.
pub fn get_tool(name: &str) -> Option<Arc<dyn Tool>> {
    TOOL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|reg| reg.get(name).cloned())
}

/// List registered tool names.
pub fn list_tools() -> Vec<String> {
    TOOL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map(|reg| reg.keys().cloned().collect())
        .unwrap_or_default()
}

// ---------------------------------
// Environment factory + registry
// ---------------------------------

/// Config-aware factory for constructing environment instances.
pub type EnvConfigFactory = Arc<dyn Fn(Option<Json>) -> Result<Box<dyn Environment>, EngineError> + Send + Sync + 'static>;

static ENV_REGISTRY: OnceLock<Mutex<HashMap<String, EnvConfigFactory>>> = OnceLock::new();

/// Register an environment factory that ignores config.
pub fn register_environment(name: &str, factory: Arc<dyn Fn() -> Box<dyn Environment> + Send + Sync + 'static>) {
    let f: EnvConfigFactory = Arc::new(move |_cfg: Option<Json>| Ok(factory())) as EnvConfigFactory;
    register_environment_with_config(name, f);
}

/// Register a config-aware environment factory under a unique name.
pub fn register_environment_with_config(name: &str, factory: EnvConfigFactory) {
    let mut reg = ENV_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("env registry poisoned");
    reg.insert(name.to_string(), factory);
}

/// Instantiate a registered environment by name with optional JSON config.
pub fn create_environment_with_config(name: &str, config: Option<Json>) -> Result<Box<dyn Environment>, EngineError> {
    let reg = ENV_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| EngineError::Internal("env registry poisoned".into()))?;
    let f = reg
        .get(name)
        .ok_or_else(|| EngineError::NotFound(format!("unsupported environment: {name}")))?;
    f(config)
}

/// Instantiate a registered environment by name with no config.
pub fn create_environment(name: &str) -> Result<Box<dyn Environment>, EngineError> {
    create_environment_with_config(name, None)
}

/// List registered environment names.
pub fn list_environments() -> Vec<String> {
    ENV_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map(|reg| reg.keys().cloned().collect())
        .unwrap_or_default()
}

// -----------------------
// Reproducibility traits
// -----------------------

/// Engines that support snapshotting and restoration.
#[async_trait]
pub trait ReproducibleEngine: Send + Sync {
    async fn serialize_engine(&self) -> Result<Json, EngineError>;
    fn engine_name(&self) -> String;
}

/// Helper to build Snapshots from a ReproducibleEngine.
pub async fn make_snapshot(engine: &dyn ReproducibleEngine, version: u32) -> Result<Snapshot, EngineError> {
    let data = engine.serialize_engine().await?;
    let engine_name = engine.engine_name();
    Ok(Snapshot { version, engine: engine_name, data })
}

// -----------------------
// Tests
// -----------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &'static str { "echo" }
        async fn call(&self, call: &ToolCall) -> Result<ToolResult, EngineError> {
            Ok(ToolResult { ok: true, payload: Some(call.args.clone()), error: None })
        }
    }

    struct NopEnv;
    #[async_trait]
    impl Environment for NopEnv {
        async fn initialize(&mut self) -> Result<Observation, EngineError> { Ok(Observation { terminated: false, truncated: false, data: Json::Null }) }
        async fn step(&mut self, _tool_calls: Vec<ToolCall>) -> Result<Observation, EngineError> { Ok(Observation { terminated: false, truncated: false, data: Json::Null }) }
        async fn checkpoint(&self) -> Result<Snapshot, EngineError> { Ok(Snapshot { version: 1, engine: "nop".into(), data: Json::Null }) }
        async fn terminate(&mut self) -> Result<Observation, EngineError> { Ok(Observation { terminated: true, truncated: false, data: Json::Null }) }
    }

    #[test]
    fn tool_registry_registers_and_lists() {
        register_tool(Arc::new(EchoTool));
        assert!(list_tools().contains(&"echo".to_string()));
        assert!(get_tool("echo").is_some());
    }

    #[test]
    fn env_registry_registers_and_lists() {
        register_environment("nop", Arc::new(|| Box::new(NopEnv)));
        assert!(list_environments().contains(&"nop".to_string()));
        // We don't invoke async methods here to avoid requiring a runtime.
        assert!(create_environment("nop").is_ok());
    }
}
