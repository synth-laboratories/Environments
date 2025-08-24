use pyo3::prelude::*;
use pyo3::types::PyModule;
use horizons_core::{Environment, ToolCall};
use serde_json::{json, Value as Json};
use tictactoe_env::{Config as TtConfig, TicTacToeEnvironment};
use sokoban_env as _sokoban_env; // ensure crate is linked
use minigrid_env as _minigrid_env; // ensure crate is linked

fn json_to_pydict(py: Python, v: &Json) -> PyResult<PyObject> {
    let s = serde_json::to_string(v).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let json = PyModule::import(py, "json")?;
    let obj = json.call_method1("loads", (s,))?;
    Ok(obj.into())
}

fn py_to_json(py: Python, obj: &PyAny) -> PyResult<Json> {
    let json = PyModule::import(py, "json")?;
    let s = json.call_method1("dumps", (obj,))?;
    let s: String = s.extract()?;
    serde_json::from_str(&s).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

fn obs_to_internal(public_data: &Json, terminated: bool, truncated: bool) -> Json {
    // Split reward fields into private; keep the rest public, and add flags.
    let mut public = match public_data {
        Json::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    // Extract rewards if present
    let mut private = serde_json::Map::new();
    if let Some(v) = public.remove("reward_last") { private.insert("reward_last".into(), v); }
    if let Some(v) = public.remove("total_reward") { private.insert("total_reward".into(), v); }
    public.insert("terminated".into(), Json::Bool(terminated));
    public.insert("truncated".into(), Json::Bool(truncated));
    json!({
        "public_observation": Json::Object(public),
        "private_observation": Json::Object(private),
    })
}

/// Minimal PyO3 module to validate the Python import path.
#[pyfunction]
fn ping() -> PyResult<&'static str> { Ok("ok") }

#[pyclass]
struct PyTicTacToeEnv {
    rt: tokio::runtime::Runtime,
    env: TicTacToeEnvironment,
}

#[pymethods]
impl PyTicTacToeEnv {
    #[new]
    fn new(agent_mark: Option<String>, opponent_minimax_prob: Option<f64>, seed: Option<u64>) -> PyResult<Self> {
        let cfg = TtConfig {
            agent_mark: agent_mark.unwrap_or_else(|| "X".into()),
            opponent_minimax_prob: opponent_minimax_prob.unwrap_or(0.0),
            seed: seed.unwrap_or(42),
        };
        let env = TicTacToeEnvironment::new(cfg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { rt, env })
    }

    fn initialize(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let obs = self.rt.block_on(self.env.initialize())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mapped = obs_to_internal(&obs.data, obs.terminated, obs.truncated);
        json_to_pydict(py, &mapped)
    }

    fn interact(&mut self, py: Python<'_>, letter: &str, number: i64) -> PyResult<PyObject> {
        let call = ToolCall { tool: "interact".into(), args: json!({"letter": letter, "number": number}) };
        let obs = self.rt.block_on(self.env.step(vec![call]))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mapped = obs_to_internal(&obs.data, obs.terminated, obs.truncated);
        json_to_pydict(py, &mapped)
    }

    fn terminate(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let obs = self.rt.block_on(self.env.terminate())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mapped = obs_to_internal(&obs.data, obs.terminated, obs.truncated);
        json_to_pydict(py, &mapped)
    }

    fn checkpoint(&self, py: Python<'_>) -> PyResult<PyObject> {
        let snap = self.rt.block_on(self.env.checkpoint())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let obj = json!({
            "version": snap.version,
            "engine": snap.engine,
            "data": snap.data,
        });
        json_to_pydict(py, &obj)
    }
}

#[pyclass]
struct PyRustEnv {
    rt: tokio::runtime::Runtime,
    env: Box<dyn Environment>,
}

#[pymethods]
impl PyRustEnv {
    fn initialize(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let obs = self.rt.block_on(self.env.initialize())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mapped = obs_to_internal(&obs.data, obs.terminated, obs.truncated);
        json_to_pydict(py, &mapped)
    }

    fn step_tool(&mut self, py: Python<'_>, tool: &str, args: &PyAny) -> PyResult<PyObject> {
        let args_json = py_to_json(py, args)?;
        let call = ToolCall { tool: tool.to_string(), args: args_json };
        let obs = self.rt.block_on(self.env.step(vec![call]))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mapped = obs_to_internal(&obs.data, obs.terminated, obs.truncated);
        json_to_pydict(py, &mapped)
    }

    fn terminate(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let obs = self.rt.block_on(self.env.terminate())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let mapped = obs_to_internal(&obs.data, obs.terminated, obs.truncated);
        json_to_pydict(py, &mapped)
    }

    fn checkpoint(&self, py: Python<'_>) -> PyResult<PyObject> {
        let snap = self.rt.block_on(self.env.checkpoint())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let obj = json!({
            "version": snap.version,
            "engine": snap.engine,
            "data": snap.data,
        });
        json_to_pydict(py, &obj)
    }
}

#[pyfunction]
fn list_rust_envs() -> PyResult<Vec<String>> {
    Ok(horizons_core::list_environments())
}

#[pyfunction]
fn create_rust_env(py: Python<'_>, name: &str, config: Option<&PyAny>) -> PyResult<Py<PyRustEnv>> {
    let cfg_json = if let Some(obj) = config { Some(py_to_json(py, obj)?) } else { None };
    let env = horizons_core::create_environment_with_config(name, cfg_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Py::new(py, PyRustEnv { rt, env })
}

#[pymodule]
fn horizons_env_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Ensure built-in Rust environments are registered
    tictactoe_env::register_default_env();
    _sokoban_env::register_default_env();
    _minigrid_env::register_default_env();
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(list_rust_envs, m)?)?;
    m.add_function(wrap_pyfunction!(create_rust_env, m)?)?;
    m.add_class::<PyTicTacToeEnv>()?;
    m.add_class::<PyRustEnv>()?;
    Ok(())
}
