use crate::level::Level;
use once_cell::sync::OnceCell;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct PresetRow {
    seed: u64,
    #[allow(dead_code)]
    #[serde(default)]
    width: Option<usize>,
    #[allow(dead_code)]
    #[serde(default)]
    height: Option<usize>,
    #[allow(dead_code)]
    #[serde(default)]
    dim_room: Option<(usize, usize)>,
    room_fixed: Vec<Vec<u8>>,
    room_state: Vec<Vec<u8>>,
}

fn flatten_2d(v: &[Vec<u8>]) -> Vec<u8> {
    let h = v.len();
    let w = if h > 0 { v[0].len() } else { 0 };
    let mut out = Vec::with_capacity(w * h);
    for row in v {
        out.extend_from_slice(row);
    }
    out
}

static PRESET_MAP: OnceCell<HashMap<u64, Level>> = OnceCell::new();

fn load_presets() -> HashMap<u64, Level> {
    let mut map = HashMap::new();
    // Embed JSONL at compile time; if missing or empty, this will just be an empty map.
    let data = include_str!("../data/prebaked_levels.jsonl");
    for (lineno, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let row: Result<PresetRow, _> = serde_json::from_str(line);
        match row {
            Ok(r) => {
                // Derive dimensions from arrays to avoid ambiguity
                let h = r.room_fixed.len();
                if h == 0 { continue; }
                let w = r.room_fixed[0].len();
                if w == 0 { continue; }
                // Basic shape checks
                if r.room_state.len() != h || r.room_state.get(0).map(|row| row.len()).unwrap_or(0) != w {
                    eprintln!("warn: preset seed {} has inconsistent shapes; skipping", r.seed);
                    continue;
                }
                let mut lvl = Level { width: w, height: h, room_fixed: Vec::new(), room_state: Vec::new() };
                lvl.room_fixed = flatten_2d(&r.room_fixed);
                lvl.room_state = flatten_2d(&r.room_state);
                map.insert(r.seed, lvl);
            }
            Err(_e) => {
                // ignore malformed lines
                eprintln!("warn: failed to parse preset line {}", lineno + 1);
            }
        }
    }
    map
}

pub fn preset_level(seed: u64) -> Option<Level> {
    let map = PRESET_MAP.get_or_init(load_presets);
    map.get(&seed).cloned()
}
