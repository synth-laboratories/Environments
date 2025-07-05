# Synth Trace Viewer

A comprehensive visualization system for exploring agent evaluation traces across multiple environments.

## Overview

The trace viewer provides two interfaces for analyzing agent behavior:
- **Streamlit App**: Interactive web interface for detailed trace analysis
- **Reflex Frontend**: Modern React-based viewer (experimental)

## Quick Start

### 1. Set Database Path
```bash
export TRACE_DB="/path/to/your/synth_eval.duckdb"
```

### 2. Run Streamlit Viewer
```bash
cd src/viewer
streamlit run streamlit_app.py --server.port 8501
```

### 3. Run Reflex Frontend (Optional)
```bash
cd src/viewer
./run_viewer.sh
```

## Architecture

### Core Components

**`streamlit_app.py`** - Main Streamlit application
- Environment-specific trace processors
- Interactive turn-by-turn visualization
- Agent reasoning display
- Raw trace data inspection

**`synth_sdk_trace_helpers.py`** - Trace processing utilities
- Synth SDK trace format parsing
- Agent reasoning extraction
- Environment state processing

**`validate_synth_sdk_trace.py`** - Trace validation
- Schema validation for trace files
- Data integrity checks
- Error reporting

### Database Layer

**`db_config.py`** - Database configuration
- Automatic path detection
- Environment variable support
- Multi-location fallback

**`db_queries.py`** - Database operations
- Environment listing
- Evaluation queries
- Trace retrieval

**`db_models.py`** - Data models
- SQLAlchemy ORM models
- Type definitions

### Frontend Components

**`frontend/viewers/`** - Environment-specific viewers
- `base.py` - Common viewer interface
- `crafter.py` - Crafter environment viewer
- `nethack.py` - NetHack environment viewer
- `minigrid.py` - MiniGrid environment viewer

**`frontend/traces_viewer/`** - Reflex application
- Modern React-based interface
- Real-time trace exploration
- Interactive visualizations

## Supported Environments

### ✅ Fully Supported
- **Crafter**: Image-based observations, achievement tracking
- **Sokoban**: Grid-based puzzles, box placement visualization

### 🔄 Partial Support
- **NetHack**: ASCII display, action mapping (simplified)
- **MiniGrid**: Grid visualization, basic action display

### 🚧 Extensible
The viewer uses a plugin-like architecture. New environments can be added by:
1. Creating processor functions in `streamlit_app.py`
2. Adding render functions for visualization
3. Implementing environment detection logic

## Usage

### Viewing Traces

1. **Select Environment**: Choose from available environments in sidebar
2. **Pick Evaluation**: Select specific evaluation run
3. **Browse Traces**: Filter and select individual traces
4. **Analyze**: Step through agent reasoning and environment responses

### Key Features

- **Agent Reasoning**: View LLM thought processes step-by-step
- **Environment State**: See state changes and observations
- **Action Analysis**: Understand agent decision-making
- **Reward Tracking**: Monitor reward signals over time
- **Raw Data Access**: Inspect full trace JSON when needed

### Filtering Options

- **Agent Reasoning**: Only show traces with recorded agent thoughts
- **Reward Threshold**: Filter by minimum reward achieved
- **Success Status**: Show only successful/failed episodes
- **Model Type**: Filter by LLM model used

## Configuration

### Environment Variables

```bash
# Database location (required)
export TRACE_DB="/path/to/synth_eval.duckdb"

# Optional: Custom port for Streamlit
export STREAMLIT_PORT=8501

# Optional: Enable debug mode
export TRACE_VIEWER_DEBUG=1
```

### Database Setup

The viewer expects a DuckDB database with evaluation data. Common locations:
- `$PROJECT_ROOT/synth_eval.duckdb`
- `$TRACE_DB` environment variable
- Current working directory

## Development

### Adding New Environments

1. **Create processor function**:
```python
def process_your_env_trace(trace_data: Dict) -> Dict:
    # Extract and process trace data
    return processed_data
```

2. **Add renderer function**:
```python
def render_your_env_trace(processed_trace: Dict):
    # Create Streamlit visualization
    pass
```

3. **Update environment detection**:
```python
if "your_env" in meta_env_name.lower():
    env_type = "your_env"
```

### Code Organization

```
src/viewer/
├── streamlit_app.py          # Main Streamlit app
├── synth_sdk_trace_helpers.py # Trace processing
├── validate_synth_sdk_trace.py # Validation
├── db_*.py                   # Database layer
├── frontend/                 # Reflex frontend
│   ├── viewers/             # Environment viewers
│   └── traces_viewer/       # Main Reflex app
└── backend/                 # API server (optional)
```

## Troubleshooting

### Common Issues

**Database not found**
- Set `TRACE_DB` environment variable
- Ensure database file exists and is readable
- Check file permissions

**No traces visible**
- Verify evaluation data is in database
- Check environment name matching
- Enable debug mode for detailed logs

**Missing agent reasoning**
- Ensure traces have `agent_compute_step` sections
- Check Synth SDK tracing configuration
- Verify trace format compliance

**Performance issues**
- Large trace files may load slowly
- Consider filtering by date/model
- Use raw data view sparingly

### Debug Mode

Enable debug logging:
```bash
export TRACE_VIEWER_DEBUG=1
streamlit run streamlit_app.py --logger.level=debug
```

## Future Enhancements

### Planned Features
- Environment plugin system (see `todos/env_specific.md`)
- Real-time trace streaming
- Comparative analysis tools
- Export functionality
- Advanced filtering options

### Known Limitations
- NetHack glyph rendering is simplified
- Image serving not implemented in backend
- Some environments have hardcoded detection logic
- Limited to DuckDB database format

## Contributing

1. Follow the environment-specific viewer pattern
2. Add comprehensive trace processing
3. Include agent reasoning extraction
4. Test with sample traces
5. Update documentation

For detailed implementation guidance, see `docs/env_contribution_guide.md`.