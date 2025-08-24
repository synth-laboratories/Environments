use std::{collections::HashMap, sync::{Arc, RwLock}, sync::atomic::{AtomicU64, Ordering}};

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::{get, post}, Json, Router};
use horizons_core::{create_environment_with_config, list_environments, Environment, EngineError, Observation, Snapshot, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Clone)]
pub struct AppState {
    store: Arc<RwLock<HashMap<String, Box<dyn Environment>>>>,
    id_ctr: Arc<AtomicU64>,
}

impl AppState {
    fn new() -> Self {
        Self { store: Arc::new(RwLock::new(HashMap::new())), id_ctr: Arc::new(AtomicU64::new(1)) }
    }
    fn next_id(&self) -> String { format!("env-{}", self.id_ctr.fetch_add(1, Ordering::Relaxed)) }
}

#[derive(Deserialize)]
pub struct InitRequest {
    pub env_type: String,
    #[serde(default)]
    pub config: Option<JsonValue>,
}

#[derive(Serialize)]
pub struct InitResponse {
    pub env_id: String,
    pub observation: Observation,
}

#[derive(Deserialize)]
pub struct StepRequest {
    pub env_id: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Deserialize)]
pub struct IdRequest { pub env_id: String }

async fn list_envs() -> impl IntoResponse { Json(list_environments()) }

#[axum::debug_handler]
async fn initialize(State(state): State<AppState>, Json(req): Json<InitRequest>) -> Result<Json<InitResponse>, (StatusCode, String)> {
    let mut env = create_environment_with_config(&req.env_type, req.config).map_err(map_engine_err)?;
    let obs = env.initialize().await.map_err(map_engine_err)?;
    let id = state.next_id();
    state.store.write().unwrap().insert(id.clone(), env);
    Ok(Json(InitResponse { env_id: id, observation: obs }))
}

#[axum::debug_handler]
async fn step(State(state): State<AppState>, Json(req): Json<StepRequest>) -> Result<Json<Observation>, (StatusCode, String)> {
    let mut guard = state.store.write().unwrap();
    let env = guard.get_mut(&req.env_id).ok_or((StatusCode::NOT_FOUND, format!("env {} not found", req.env_id)))?;
    let obs = env.step(req.tool_calls).await.map_err(map_engine_err)?;
    Ok(Json(obs))
}

#[axum::debug_handler]
async fn checkpoint(State(state): State<AppState>, Json(req): Json<IdRequest>) -> Result<Json<Snapshot>, (StatusCode, String)> {
    let guard = state.store.read().unwrap();
    let env = guard.get(&req.env_id).ok_or((StatusCode::NOT_FOUND, format!("env {} not found", req.env_id)))?;
    let snap = env.checkpoint().await.map_err(map_engine_err)?;
    Ok(Json(snap))
}

#[axum::debug_handler]
async fn terminate(State(state): State<AppState>, Json(req): Json<IdRequest>) -> Result<Json<Observation>, (StatusCode, String)> {
    let mut guard = state.store.write().unwrap();
    let mut env = guard.remove(&req.env_id).ok_or((StatusCode::NOT_FOUND, format!("env {} not found", req.env_id)))?;
    let obs = env.terminate().await.map_err(map_engine_err)?;
    Ok(Json(obs))
}

fn map_engine_err(err: EngineError) -> (StatusCode, String) {
    match err {
        EngineError::Validation(s) => (StatusCode::BAD_REQUEST, s),
        EngineError::NotFound(s) => (StatusCode::NOT_FOUND, s),
        EngineError::Internal(s) => (StatusCode::INTERNAL_SERVER_ERROR, s),
    }
}

pub fn make_app() -> Router {
    let state = AppState::new();
    Router::new()
        .route("/envs", get(list_envs))
        .route("/initialize", post(initialize))
        .route("/step", post(step))
        .route("/checkpoint", post(checkpoint))
        .route("/terminate", post(terminate))
        .with_state(state)
}
