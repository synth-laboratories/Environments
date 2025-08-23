# Crafter Classic Agent Demos

This directory contains the core agent implementation and evaluation framework for testing AI agents on the Crafter Classic environment.

## 🎯 Quick Start

**For development and testing:**
```bash
# Run evaluation using the framework directly
python -c "
from crafter_evaluation_framework import run_crafter_eval
import asyncio
asyncio.run(run_crafter_eval(['gpt-4o-mini'], ['easy'], 3, 30))
"
```

**For comprehensive evaluation with traces (legacy):**
```bash
# Use files in old/ directory
python old/crafter_trace_evaluation.py --models gpt-4o-mini --difficulties easy --num-trajectories 5
```

## 📁 Files Overview

### Core Agent Implementation
- **`crafter_react_agent.py`** (34KB, 910 lines) - **Main ReAct agent implementation**
  - Core agent logic using ReAct (Reasoning + Acting) pattern
  - Handles observation processing, action selection, and tool usage
  - Includes evaluation functions and test utilities
  - **Use for:** Understanding agent behavior, debugging, extending agent capabilities

### Evaluation Infrastructure
- **`crafter_evaluation_framework.py`** (48KB, 1293 lines) - **Core evaluation framework**
  - Standardized metrics: Hafner scores, BALROG scores, achievement tracking
  - Comprehensive data collection and analysis
  - **Use for:** Understanding evaluation metrics, extending evaluation capabilities

### Configuration
- **`crafter_evaluation_config.toml`** (460B, 24 lines) - **Default evaluation configuration**
  - Standard settings for evaluation runs
  - Models, difficulties, parallel execution settings
  - **Use for:** Configuring evaluation parameters

### Legacy Files (Moved to `old/` directory)
- **`old/`** - Contains moved files for reference:
  - `crafter_quick_evaluation.py` - Simple evaluation wrapper
  - `crafter_trace_evaluation.py` - Comprehensive evaluation with traces
  - `crafter_comprehensive_evaluation.py` - Alternative evaluation script
  - `crafter_evaluation_browser.py` - Evaluation browser utility
  - `test_crafter_react_agent.py` - Large test file (to be converted to pytest)

## 🚀 Usage Examples

### 1. Direct Framework Usage
```bash
# Run evaluation using the framework directly
python -c "
from crafter_evaluation_framework import run_crafter_eval
import asyncio
asyncio.run(run_crafter_eval(['gpt-4o-mini'], ['easy'], 3, 30))
"
```

### 2. Using Legacy Scripts (if needed)
```bash
# Use files from old/ directory
python old/crafter_trace_evaluation.py \
  --models gpt-4o-mini gpt-4.1-nano \
  --difficulties easy hard \
  --num-trajectories 10 \
  --max-turns 50
```

# View specific evaluation
python crafter_evaluation_browser.py --run-id run_20240704_143022

# View latest evaluation
python crafter_evaluation_browser.py --latest
```

### 4. Comprehensive Evaluation
```bash
# Full evaluation with all options
python crafter_comprehensive_evaluation.py \
  --models gpt-4o-mini gpt-4.1-nano \
  --difficulties easy hard \
  --num-trajectories 20 \
  --max-turns 100 \
  --output-dir custom_eval_dir
```

## 📊 Evaluation Metrics

The evaluation framework computes several key metrics:

### Achievement Tracking
- **Hafner Score**: Weighted achievement score based on Crafter paper
- **BALROG Score**: Alternative weighted scoring system
- **Achievement Count**: Raw number of achievements unlocked
- **Achievement Rate**: Achievements per episode

### Performance Metrics
- **Success Rate**: Percentage of episodes achieving goals
- **Average Reward**: Mean reward across episodes
- **Episode Length**: Average number of turns per episode
- **Completion Time**: Time to complete episodes

### Behavioral Analysis
- **Action Distribution**: Frequency of different actions
- **Tool Usage**: How often each tool is used
- **Error Rate**: Frequency of invalid actions
- **Reasoning Quality**: Analysis of agent's reasoning steps

## 🔧 Configuration Options

### Model Configuration
- `models`: List of model names to evaluate
- `difficulties`: Environment difficulty levels (`easy`, `hard`)
- `num_trajectories`: Number of episodes per condition
- `max_turns`: Maximum turns per episode

### Execution Settings
- `parallel_episodes`: Enable parallel execution
- `timeout_seconds`: Timeout for each episode
- `capture_images`: Save screenshots during evaluation
- `launch_viewer`: Auto-launch viewer after evaluation

### Output Settings
- `output_dir`: Custom output directory
- `show_progress_bars`: Display progress during evaluation
- `show_detailed_logging`: Verbose logging
- `show_final_table`: Summary table at end

## 📂 Output Structure

Evaluations create the following structure:
```
src/evals/crafter/run_YYYYMMDD_HHMMSS/
├── evaluation_summary.json     # Aggregate results and metadata
├── traces/                     # Individual episode traces
│   ├── trace_uuid_1.json      # Trace for episode 1
│   ├── trace_uuid_2.json      # Trace for episode 2
│   └── ...
└── viewer/                     # Standalone trace viewer
    ├── index.html             # Viewer interface
    ├── traces.js              # Trace data
    └── viewer.js              # Viewer logic
```

## 🧪 Development Workflow

### 1. Agent Development
1. Modify `crafter_react_agent.py` to test new agent behaviors
2. Run `crafter_quick_evaluation.py` for rapid iteration
3. Use `crafter_evaluation_framework.py` for detailed metrics

### 2. Evaluation Setup
1. Configure `crafter_evaluation_config.toml` with desired settings
2. Run `crafter_trace_evaluation.py` to generate traces
3. Use `crafter_evaluation_browser.py` to review results

### 3. Analysis and Debugging
1. Browse evaluations with `crafter_evaluation_browser.py`
2. Launch trace viewer to examine individual episodes
3. Analyze agent reasoning and environment interactions

## 🎮 Environment Details

**Crafter Classic** is a 2D survival game where agents must:
- Gather resources (wood, stone, food)
- Craft tools and items
- Build shelter
- Survive environmental challenges
- Unlock achievements through strategic play

**Key Features:**
- Rich visual environment with pixel art graphics
- Complex action space with 17 different actions
- Achievement system with 22 different achievements
- Multiple difficulty levels
- Comprehensive observation space including inventory, health, and environment state

## 🔍 Troubleshooting

### Common Issues

**"No evaluations found"**
- Check that evaluations are saved to `src/evals/crafter/`
- Ensure `evaluation_summary.json` exists in run directories

**"Viewer files not found"**
- Verify that `crafter_trace_evaluation.py` was used to generate traces
- Check that `viewer/` directory exists in the evaluation run

**"Import errors"**
- Ensure you're running from the correct directory
- Check that all dependencies are installed
- Verify Python path includes the synth_env package

**"Timeout errors"**
- Increase `timeout_seconds` in configuration
- Reduce `max_turns` for faster episodes
- Check model API rate limits

### Performance Tips

- Use `parallel_episodes=true` for faster evaluation
- Reduce `num_trajectories` for quicker testing
- Set `capture_images=false` to save disk space
- Use `gpt-4o-mini` for faster, cheaper evaluation

## 📚 Related Documentation

- [Crafter Environment Guide](../README.md) - Environment setup and usage
- [Trace Viewer Documentation](../../../../viewer/README.md) - Viewing and analyzing traces
- [Evaluation Framework](../../../docs/evaluation_framework.md) - General evaluation principles
- [Agent Development Guide](../../../docs/agent_development.md) - Building new agents

## 🤝 Contributing

When adding new evaluation scripts or modifying existing ones:

1. **Follow naming conventions**: Use descriptive names like `crafter_*_evaluation.py`
2. **Update this README**: Document new files and their purposes
3. **Include configuration options**: Support command-line arguments and config files
4. **Generate traces**: Ensure compatibility with the trace viewer system
5. **Add error handling**: Graceful handling of failures and edge cases
6. **Test thoroughly**: Verify with different models and configurations

## 📈 Future Enhancements

Planned improvements for the evaluation system:

- **Multi-agent evaluation**: Support for multiple agents in the same environment
- **Curriculum learning**: Progressive difficulty evaluation
- **Comparative analysis**: Direct comparison between different agents
- **Real-time monitoring**: Live evaluation progress tracking
- **Automated reporting**: Generated reports with insights and recommendations