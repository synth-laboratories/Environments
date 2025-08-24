use std::net::SocketAddr;
use env_service::make_app;
use tictactoe_env::register_default_env as register_ttt;
use sokoban_env::register_default_env as register_sokoban;

#[derive(Clone)]
struct AppState {
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
struct InitRequest {
    env_type: String,
    #[serde(default)]
    config: Option<JsonValue>,
}

#[derive(Serialize)]
struct InitResponse {
    env_id: String,
    observation: Observation,
}

#[derive(Deserialize)]
struct StepRequest {
    env_id: String,
    tool_calls: Vec<ToolCall>,
}

#[derive(Deserialize)]
struct IdRequest { env_id: String }

async fn list_envs() -> impl IntoResponse {
    Json(list_environments())
}

async fn initialize(State(state): State<AppState>, Json(req): Json<InitRequest>) -> Result<Json<InitResponse>, (StatusCode, String)> {
    // Generic config-aware env creation
    let env_res: Result<Box<dyn Environment>, EngineError> = create_environment_with_config(&req.env_type, req.config);

    let mut env = env_res.map_err(map_engine_err)?;
    let obs = env.initialize().await.map_err(map_engine_err)?;
    let id = state.next_id();
    state.store.write().unwrap().insert(id.clone(), env);
    Ok(Json(InitResponse { env_id: id, observation: obs }))
}

async fn step(State(state): State<AppState>, Json(req): Json<StepRequest>) -> Result<Json<Observation>, (StatusCode, String)> {
    let mut guard = state.store.write().unwrap();
    let env = guard.get_mut(&req.env_id).ok_or((StatusCode::NOT_FOUND, format!("env {} not found", req.env_id)))?;
    let obs = env.step(req.tool_calls).await.map_err(map_engine_err)?;
    Ok(Json(obs))
}

async fn checkpoint(State(state): State<AppState>, Json(req): Json<IdRequest>) -> Result<Json<Snapshot>, (StatusCode, String)> {
    let guard = state.store.read().unwrap();
    let env = guard.get(&req.env_id).ok_or((StatusCode::NOT_FOUND, format!("env {} not found", req.env_id)))?;
    let snap = env.checkpoint().await.map_err(map_engine_err)?;
    Ok(Json(snap))
}

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

#[tokio::main]
async fn main() {
    // Pre-register environments for /envs and factory-based init
    register_ttt();
    register_sokoban();
    let app = make_app();

    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    println!("Environment service listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app).await.unwrap();
}
