# TicTacToe Environment

A complete TicTacToe game environment for reinforcement learning research and experimentation within the Horizons AI framework.

## Overview

This environment provides a classic 3x3 TicTacToe game where agents can learn to play optimally against opponents. The environment supports various starting positions, opponent strategies, and comprehensive game state tracking.

## Features

- **Complete Game Logic**: Full TicTacToe rules implementation
- **Multiple Starting Positions**: Support for games starting with 0-3 pre-moves
- **Tool-Based Interaction**: Uses the Horizons tool system for game actions
- **Stateful Environment**: Implements the complete StatefulEnvironment interface
- **Reproducible Games**: Support for deterministic game replay
- **Reward System**: Configurable reward components for different game outcomes
- **Observation System**: Rich observation format with board visualization

## Game Rules

- **Board**: 3x3 grid with cells labeled A1-A3, B1-B3, C1-C3
- **Players**: X (first player) and O (second player)
- **Objective**: Get three marks in a row (horizontal, vertical, or diagonal)
- **Game End**: Win, loss, or draw (board full)

## Installation and Setup

The TicTacToe environment is included with the Horizons AI framework and requires no additional setup.

## Basic Usage

### Creating and Initializing

```python
from horizons.environments.examples.tictactoe.environment import TicTacToeEnvironment
from horizons.environments.tasks.core import Impetus, Intent, TaskInstance, TaskInstanceMetadata
from uuid import uuid4

# Create a task instance
task = TaskInstance(
    id=uuid4(),
    impetus=Impetus("Play TicTacToe and win the game!"),
    intent=Intent(rubric={"goal": "win the game"}),
    metadata=TaskInstanceMetadata(),
    is_reproducible=True,
    initial_engine_snapshot=None
)

# Create environment
env = TicTacToeEnvironment(task)

# Initialize (starts a new game)
obs = await env.initialize()
```

### Making Moves

```python
from horizons.environments.environment.tools import EnvToolCall

# Make a move using coordinate notation
tool_call = EnvToolCall(
    name="interact",
    args={"letter": "A", "number": 1}  # Place mark at A1
)

# Validate and execute the move
env.validate_tool_calls([tool_call])
obs = await env.step([tool_call])
```

### Checking Game State

```python
# Access current game state
board_text = obs.public_observation['board_text']  # ASCII board representation
current_player = obs.public_observation['current_player']  # "X" or "O"
move_count = obs.public_observation['move_count']  # Number of moves made
winner = obs.public_observation['winner']  # None, "X", "O", or "draw"
terminated = obs.public_observation['terminated']  # True if game over
```

## Action Space

The environment uses a tool-based action system with a single `interact` tool:

### Tool: `interact`

**Purpose**: Place a mark on the TicTacToe board

**Parameters**:
- `letter`: Board column (string: "A", "B", or "C")
- `number`: Board row (integer: 1, 2, or 3)

**Examples**:
- `{"letter": "A", "number": 1}` → Place mark at A1
- `{"letter": "B", "number": 3}` → Place mark at B3
- `{"letter": "C", "number": 2}` → Place mark at C2

**Validation**:
- Letter must be A, B, or C
- Number must be 1, 2, or 3
- Cell must be empty
- Game must not be terminated

## Observation Space

The environment provides rich observations through the InternalObservation system:

### Public Observation
```python
{
    "board_text": "  A B C\n1 X O  \n2   X  \n3 O X  ",
    "current_player": "X",
    "move_count": 5,
    "last_move": "B2",
    "winner": None,
    "terminated": False
}
```

### Private Observation
```python
{
    "reward_last": 0.0,
    "total_reward": 0.0,
    "terminated": False,
    "truncated": False
}
```

## Reward System

The environment includes configurable reward components:

### Reward Components
1. **Win Component**: +1.0 for winning, -1.0 for losing
2. **Draw Component**: 0.0 for draws (configurable)
3. **Illegal Move Component**: -1.0 for invalid moves

### Reward Configuration
```python
# Configure rewards for specific player
win_component = TicTacToeWinComponent(player_mark="X")
reward_stack = RewardStack([win_component, ...])
```

## Task System

The environment supports the Horizons task system for creating varied game scenarios:

### Creating Task Instances
```python
from horizons.environments.examples.tictactoe.taskset import create_tictactoe_taskset

# Generate diverse TicTacToe scenarios
taskset = await create_tictactoe_taskset()

# Access individual task instances
for task_instance in taskset.instances:
    env = TicTacToeEnvironment(task_instance)
    # Each instance has different starting positions
```

### Task Instance Metadata
```python
metadata = task_instance.metadata
print(f"Starting player: {metadata.starting_player}")
print(f"Opening moves: {metadata.opening_moves}")
print(f"Position complexity: {metadata.position_complexity}")
```

## Examples and Demos

### Running the Demo
```bash
cd src/horizons/environments/examples/tictactoe
python demo.py
```

### Agent Integration
```python
# Example of using with an agent
async def play_game(env):
    obs = await env.initialize()

    while not obs.public_observation['terminated']:
        # Agent logic here
        move = agent.choose_move(obs.public_observation)

        tool_call = EnvToolCall(
            name="interact",
            args=move  # {"letter": "...", "number": ...}
        )

        obs = await env.step([tool_call])

    return obs.public_observation['winner']
```

## Testing

The environment includes comprehensive unit tests:

### Running Tests
```bash
# Run all TicTacToe tests
python -m pytest src/horizons/environments/examples/tictactoe/units/

# Run specific test
python -m pytest src/horizons/environments/examples/tictactoe/units/test_tictactoe_basic.py
```

### Test Coverage
- Environment initialization and cleanup
- Valid and invalid move handling
- Game completion scenarios
- Tool validation and execution
- Checkpoint and state management

## Advanced Features

### Reproducibility
The environment supports deterministic game replay through the IReproducibleEngine interface.

### Checkpoint System
```python
# Create game checkpoint
checkpoint_obs = await env.checkpoint()

# Checkpoint contains full game state for restoration
game_state = checkpoint_obs.private_observation
```

### Custom Observations
```python
# Use custom observation callables
custom_obs = MyCustomTicTacToeObservation()
env = TicTacToeEnvironment(
    task_instance,
    custom_step_obs=custom_obs
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all Horizons dependencies are installed
2. **Validation Errors**: Check coordinate format (A-C, 1-3)
3. **Game State Issues**: Verify game hasn't terminated before making moves

### Debug Information
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Access internal engine state
engine_state = env.engine.get_current_states_for_observation()
```

## Contributing

When contributing to the TicTacToe environment:

1. Follow the existing code patterns and style
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure backward compatibility

## Architecture

The TicTacToe environment follows the standard Horizons architecture:

- **Environment**: Main interface implementing StatefulEnvironment
- **Engine**: Game logic implementing StatefulEngine and IReproducibleEngine
- **Tools**: Action interface through AbstractTool implementation
- **Tasks**: Scenario generation through TaskInstance system
- **Rewards**: Outcome evaluation through RewardComponent system

This modular design allows for easy extension and customization while maintaining compatibility with the broader Horizons framework.
