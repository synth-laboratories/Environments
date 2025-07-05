# Synth-Env Environment Contribution Guide

This guide provides a step-by-step process for implementing new environments in the synth-env framework.

## Overview

Each environment consists of several key components:
1. Engine - Core game/task logic and state management
2. Environment - Wrapper that provides the standardized interface
3. TaskSet - Task instance generation and configuration
4. Agent Demos - Example agents (typically ReAct) that solve tasks
5. Unit Tests - Comprehensive testing of all components

## Directory Structure

Create the following structure under `src/synth_env/examples/your_env/`:

```
your_env/
├── __init__.py                    # Module init (can be minimal)
├── engine.py                      # Core logic + state management
├── environment.py                 # StatefulEnvironment wrapper
├── taskset.py                     # Task/TaskInstance generator
├── agent_demos/
│   └── test_synth_react.py       # ReAct agent evaluation
└── units/                         # Unit tests
    ├── test_your_env_engine.py
    ├── test_your_env_environment.py
    └── test_your_env_taskset.py
```

## Step-by-Step Implementation Guide

### Step 1: Define Data Models (engine.py)

Start by defining your state representations:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class YourEnvPublicState:
    """State visible to the agent"""
    # Game/task state fields
    terminated: bool
    
    def diff(self, prev_state: "YourEnvPublicState") -> Dict[str, Any]:
        """Track changes between states"""
        differences = {}
        # Add changed fields to differences dict
        return differences

@dataclass
class YourEnvPrivateState:
    """Internal state (rewards, termination flags)"""
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool
    
    def diff(self, prev_state: "YourEnvPrivateState") -> Dict[str, Any]:
        # Similar to public state diff
        pass

@dataclass
class YourEnvEngineSnapshot(StatefulEngineSnapshot):
    """Serialization container"""
    task_instance_dict: Dict
    engine_snapshot: Dict
```

### Step 2: Implement Engine Class (engine.py)

```python
from synth_env.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_env.reproducibility.core import IReproducibleEngine
from synth_env.environment.rewards.core import RewardStack, RewardComponent
from synth_env.tasks.core import TaskInstance

class YourEnvEngine(StatefulEngine, IReproducibleEngine):
    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self.reward_stack = RewardStack([
            # Add your reward components
        ])
        # Initialize state variables
        
    async def _reset_engine(self, *, seed: int | None = None) -> Tuple[YourEnvPrivateState, YourEnvPublicState]:
        """Reset to initial state"""
        # Reset all state variables
        # Return initial private and public states
        
    async def _step_engine(self, action: Any) -> Tuple[YourEnvPrivateState, YourEnvPublicState]:
        """Execute one step/action"""
        # Validate action
        # Update state
        # Check termination conditions
        # Calculate rewards via self.reward_stack.step_reward()
        # Return new states
        
    async def _serialize_engine(self) -> YourEnvEngineSnapshot:
        """Serialize current state"""
        return YourEnvEngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            engine_snapshot={
                # All state variables
            }
        )
        
    @classmethod
    async def _deserialize_engine(cls, snapshot: YourEnvEngineSnapshot) -> "YourEnvEngine":
        """Restore from serialized state"""
        # Recreate engine from snapshot
        
    def get_current_states_for_observation(self) -> Tuple[YourEnvPrivateState, YourEnvPublicState]:
        """Get current states without advancing"""
        # Return current state snapshots
```

### Step 3: Create Reward Components (engine.py)

```python
class YourEnvRewardComponent(RewardComponent):
    async def score(self, state: YourEnvPublicState, action: Any) -> float:
        """Calculate reward for this component"""
        # Return reward value based on state/action
```

### Step 4: Define Observation Callables (engine.py)

```python
from synth_env.environment.shared_engine import GetObservationCallable, InternalObservation

class YourEnvObservationCallable(GetObservationCallable):
    async def get_observation(self, pub: YourEnvPublicState, priv: YourEnvPrivateState) -> InternalObservation:
        observation: InternalObservation = {
            # Key fields agent needs to see
            "terminated": pub.terminated,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward
        }
        return observation

class YourEnvCheckpointObservationCallable(GetObservationCallable):
    async def get_observation(self, pub: YourEnvPublicState, priv: YourEnvPrivateState) -> InternalObservation:
        observation: InternalObservation = {
            # Final/checkpoint observation fields
            "total_reward": priv.total_reward,
            "terminated": pub.terminated
        }
        return observation
```

### Step 5: Create Environment Wrapper (environment.py)

```python
from synth_env.stateful.core import StatefulEnvironment
from synth_env.reproducibility.core import ReproducibleEnvironment
from synth_env.environment.tools import AbstractTool, EnvToolCall, ToolResult
from synth_env.environment.shared_engine import GetObservationCallable, InternalObservation
from pydantic import BaseModel

class YourEnvActionInput(BaseModel):
    """Pydantic model for action validation"""
    action: str  # Or appropriate type

class YourEnvInteractTool(AbstractTool):
    name = "interact"
    description = "Perform an action in the environment"
    call_schema = YourEnvActionInput
    result_schema = ToolResult
    
    def __init__(self, engine: YourEnvEngine):
        self.engine = engine
        
    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            action = call.args.get("action")
            if not action:
                return ToolResult(
                    ok=False,
                    error="No action provided",
                    payload={}
                )
            
            private_state, public_state = await self.engine._step_engine(action)
            
            return ToolResult(
                ok=True,
                payload={
                    "public_state": public_state,
                    "private_state": private_state
                }
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                error=str(e),
                payload={}
            )

class YourEnvironment(StatefulEnvironment, ReproducibleEnvironment[YourEnvEngine]):
    def __init__(self, task_instance: TaskInstance, 
                 custom_step_obs: Optional[GetObservationCallable] = None,
                 custom_ckpt_obs: Optional[GetObservationCallable] = None):
        self.name = "YourEnvironment"
        self.task_instance = task_instance
        self.custom_step_observation_callable = custom_step_obs or YourEnvObservationCallable()
        self.custom_checkpoint_observation_callable = custom_ckpt_obs or YourEnvCheckpointObservationCallable()
        self.engine = YourEnvEngine(task_instance)
        self._interact_tool = YourEnvInteractTool(self.engine)
        
    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub, self.custom_step_observation_callable)
        
    async def step(self, tool_calls) -> InternalObservation:
        validated_call = self.validate_tool_calls(tool_calls)
        result = await self._interact_tool(validated_call)
        
        if result.ok:
            priv = result.payload["private_state"]
            pub = result.payload["public_state"]
            return await self._to_observation(priv, pub, self.custom_step_observation_callable)
        else:
            priv, pub = self.engine.get_current_states_for_observation()
            return await self._to_observation(
                priv, pub, self.custom_step_observation_callable,
                extra_obs={"error": result.error}
            )
            
    # Implement remaining required methods:
    # - checkpoint()
    # - terminate()
    # - validate_tool_calls()
    # - _to_observation()
    # - _serialize_engine()
    # - _deserialize_engine()
```

### Step 6: Create TaskSet (taskset.py)

```python
from uuid import uuid4
from dataclasses import dataclass
from synth_env.tasks.core import (
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    Impetus,
    Intent,
    SplitInfo
)

@dataclass
class YourEnvTaskInstanceMetadata(TaskInstanceMetadata):
    """Task-specific metadata"""
    difficulty: str
    # Other relevant fields

@dataclass
class YourEnvTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        # Implement serialization
        
    @classmethod
    async def deserialize(cls, data: dict) -> "YourEnvTaskInstance":
        # Implement deserialization

async def create_your_env_taskset() -> TaskInstanceSet:
    """Generate task instances"""
    instances = []
    
    # Generate diverse task instances
    for config in task_configs:
        impetus = Impetus(
            instructions="Clear instructions for the task"
        )
        
        intent = Intent(
            rubric={"goal": "What success looks like"},
            gold_trajectories=None,
            gold_state_diff={}
        )
        
        metadata = YourEnvTaskInstanceMetadata(
            # Fill in metadata
        )
        
        instance = YourEnvTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None
        )
        instances.append(instance)
    
    # Define splits
    val_ids = {inst.id for inst in instances if some_condition}
    test_ids = {inst.id for inst in instances if other_condition}
    
    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True
    )
    
    return TaskInstanceSet(
        name="YourEnv TaskSet",
        description="Description of your taskset",
        instances=instances,
        split_info=split_info
    )

# Module-level export
taskset = create_your_env_taskset
```

### Step 7: Create ReAct Agent Demo (agent_demos/test_synth_react.py)

```python
import asyncio
import json
from typing import Dict, Any, List
from pydantic import BaseModel
from synth_ai.zyk import LM

class YourEnvReActAgent:
    def __init__(self, llm, max_turns: int = 10):
        self.llm = llm
        self.max_turns = max_turns
        self.history = []
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "your_env_interact",
                    "description": "Perform action in environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to take"
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "End when task is complete",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for LLM"""
        # Convert observation dict to readable string
        
    async def decide(self, obs: str) -> Dict[str, Any]:
        """Get LLM decision"""
        response = await self.llm.respond_async(
            system_message="You are playing/solving...",
            user_message=obs,
            tools=self.tools
        )
        
        # Parse tool calls from response
        # Return action dict
        
    async def run_episode(self, env: YourEnvironment) -> Dict[str, Any]:
        """Run one episode"""
        obs = await env.initialize()
        
        for turn in range(self.max_turns):
            formatted_obs = self._format_observation(obs)
            action = await self.decide(formatted_obs)
            
            if action["name"] == "terminate":
                break
                
            obs = await env.step({"action": action["parameters"]["action"]})
            
            if obs.get("terminated", False):
                break
                
        return {
            "success": obs.get("success", False),
            "turns": turn + 1,
            "final_reward": obs.get("total_reward", 0.0)
        }

async def eval_react_your_env(model_name: str = "gpt-4.1-mini") -> List[Dict[str, Any]]:
    """Evaluate ReAct agent"""
    taskset = await create_your_env_taskset()
    llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.7)
    agent = YourEnvReActAgent(llm)
    
    results = []
    for instance in taskset.instances[:10]:
        env = YourEnvironment(instance)
        result = await agent.run_episode(env)
        results.append(result)
        
    return results

if __name__ == "__main__":
    asyncio.run(eval_react_your_env())
```

### Step 8: Write Unit Tests

Create comprehensive tests for each component:

#### test_your_env_engine.py
- Test state initialization
- Test valid/invalid actions
- Test reward calculations
- Test win/termination conditions
- Test serialization/deserialization
- Test state diffs

#### test_your_env_environment.py
- Test environment initialization
- Test step functionality
- Test tool validation
- Test checkpoint/terminate
- Test full episodes

#### test_your_env_taskset.py
- Test task generation
- Test metadata validity
- Test splits
- Test serialization

### Step 9: Register in Service (app.py)

Add to `src/synth_env/service/app.py`:

```python
import synth_env.examples.your_env.environment as ye
register_environment("YourEnvironment", ye.YourEnvironment)
```

### Step 10: Type Checking and Testing

1. Run type checker:
```bash
uvx ty check src/synth_env/examples/your_env/
```

2. Run unit tests:
```bash
PYTHONPATH=src python -m pytest src/synth_env/examples/your_env/units/ -v
```

3. Run agent demo:
```bash
cd src/synth_env/examples/your_env && python agent_demos/test_synth_react.py
```

## Important Conventions and Rules

### Import Style
- Use `from __future__ import annotations` at the top
- Use absolute imports from synth_env
- Import typing annotations as needed

### Type Annotations
- All functions should have type hints
- Use `InternalObservation` for observation returns
- Intent.rubric is `Dict[str, Any]` (usually {"goal": "..."})
- Use `Optional` for nullable fields

### Async/Await
- All interface methods must be async
- Use `await` when calling other async methods
- Engine methods like `_step_engine` must be async

### State Management
- Public state = what agent sees
- Private state = internal bookkeeping
- Always implement `diff()` methods
- States should be immutable (use copies)

### Reward System
- Use RewardStack with multiple RewardComponents
- Call `reward_stack.step_reward(state, action)`
- Components should be focused on one aspect

### Tool System
- Tools must inherit from AbstractTool
- Return ToolResult with ok/error/payload
- Validate inputs with Pydantic models
- Handle multiple input formats in validate_tool_calls()

### Observation System
- Observations are Dict[str, Any]
- Separate callables for step vs checkpoint
- Type hint as InternalObservation

### Testing
- Test edge cases and error conditions
- Test serialization round-trips
- Use pytest fixtures for common setup
- Aim for >90% coverage

### Common Pitfalls to Avoid
1. Don't use `name` field in ToolResult (it doesn't exist)
2. Remember to mark states as terminated in both public and private
3. Always validate actions before applying them
4. Handle pre-moves/initial state from task metadata
5. Use dataclasses for all state objects
6. Don't forget to implement all abstract methods

## Example Environments to Reference

- **TicTacToe**: Simple turn-based game with clear win conditions
- **Sokoban**: Grid-based puzzle with complex state
- **Verilog**: Code generation with compilation/testing
- **CrafterClassic**: Real-time game with image observations
- **Math**: Text-based problem solving

Each demonstrates different patterns and complexity levels.

## Debugging Tips

1. Start simple - get basic step/reset working first
2. Use print statements in engine to debug state transitions
3. Write unit tests as you go
4. Check type errors early with `uvx ty check`
5. Test serialization/deserialization thoroughly
6. Verify observations contain all needed info

## Final Checklist

- [ ] All files created with proper structure
- [ ] Engine implements all required methods
- [ ] Environment wraps engine correctly
- [ ] TaskSet generates valid instances
- [ ] Agent demo runs successfully
- [ ] Unit tests pass with good coverage
- [ ] Type checker reports no errors
- [ ] Registered in service app.py
- [ ] Documentation/comments added where helpful

Following this guide should result in a well-integrated, fully functional environment that works seamlessly with the synth-env framework!

## Creating Evaluation Scripts with Trace Saving

### Overview

To integrate with the trace viewer system, your evaluation scripts need to:
1. Save traces in the correct format using Synth SDK
2. Store traces in the expected directory structure
3. Generate evaluation summaries with proper metadata
4. Include agent reasoning steps (not just environment interactions)

### Step 11: Create Evaluation Script with Synth SDK Tracing

Create `agent_demos/run_evaluation.py`:

```python
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from synth_sdk.tracing.decorators import trace_calls
from synth_sdk.tracing.core import get_tracer
from synth_sdk.tracing.events import (
    AgentComputeInput, AgentComputeOutput, 
    EnvironmentComputeInput, EnvironmentComputeOutput,
    MessageInput, MessageOutput
)

# Your environment imports
from ..environment import YourEnvironment
from ..taskset import create_your_env_taskset

class YourEnvEvaluationAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.llm = LM(model_name=model_name, temperature=0.7)
        
    @trace_calls
    async def run_episode(self, env: YourEnvironment, task_instance: TaskInstance) -> Dict[str, Any]:
        """Run one episode with full tracing"""
        tracer = get_tracer()
        
        # Initialize environment
        obs = await env.initialize()
        
        for turn in range(50):  # Max turns
            # === AGENT REASONING STEP ===
            # This is crucial - we need to trace the agent's thinking process
            agent_input = AgentComputeInput(
                messages=[
                    MessageInput(
                        role="system",
                        content="You are solving a puzzle. Think step by step."
                    ),
                    MessageInput(
                        role="user", 
                        content=self._format_observation(obs)
                    )
                ]
            )
            
            # Get LLM response with reasoning
            response = await self.llm.respond_async(
                system_message="You are solving a puzzle. Think step by step.",
                user_message=self._format_observation(obs),
                tools=self._get_tools()
            )
            
            # Parse action from response
            action = self._parse_action(response)
            
            # Log agent reasoning
            agent_output = AgentComputeOutput(
                messages=[
                    MessageOutput(
                        role="assistant",
                        content=response.get("content", ""),
                        tool_calls=response.get("tool_calls", [])
                    )
                ]
            )
            
            tracer.log_agent_compute_step(agent_input, agent_output)
            
            # === ENVIRONMENT STEP ===
            env_input = EnvironmentComputeInput(
                action=action,
                turn=turn
            )
            
            # Execute action in environment
            obs = await env.step({"action": action})
            
            env_output = EnvironmentComputeOutput(
                observation=obs,
                reward=obs.get("reward_last", 0.0),
                terminated=obs.get("terminated", False)
            )
            
            tracer.log_environment_compute_step(env_input, env_output)
            
            if obs.get("terminated", False):
                break
                
        return {
            "success": obs.get("success", False),
            "turns": turn + 1,
            "final_reward": obs.get("total_reward", 0.0),
            "trace_id": tracer.get_trace_id()
        }
    
    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for LLM"""
        # Convert observation to readable string
        pass
    
    def _get_tools(self) -> List[Dict]:
        """Get available tools for LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Take an action in the environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "Action to take"}
                        },
                        "required": ["action"]
                    }
                }
            }
        ]
    
    def _parse_action(self, response: Dict) -> str:
        """Parse action from LLM response"""
        # Extract action from tool calls or content
        pass

async def run_evaluation(
    model_name: str = "gpt-4o-mini",
    difficulty: str = "easy",
    num_episodes: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """Run full evaluation with trace saving"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"src/evals/your_env/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    
    # Create taskset
    taskset = await create_your_env_taskset()
    agent = YourEnvEvaluationAgent(model_name)
    
    results = []
    successful_episodes = 0
    total_reward = 0.0
    total_achievements = 0
    
    for i, task_instance in enumerate(taskset.instances[:num_episodes]):
        print(f"Running episode {i+1}/{num_episodes}")
        
        # Create environment
        env = YourEnvironment(task_instance)
        
        # Run episode with tracing
        result = await agent.run_episode(env, task_instance)
        
        # Save trace to file
        tracer = get_tracer()
        trace_data = tracer.get_trace_data()
        
        trace_file = traces_dir / f"{result['trace_id']}.json"
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        # Collect metrics
        if result["success"]:
            successful_episodes += 1
        total_reward += result["final_reward"]
        
        results.append({
            "episode": i + 1,
            "success": result["success"],
            "turns": result["turns"],
            "final_reward": result["final_reward"],
            "trace_id": result["trace_id"],
            "difficulty": difficulty,
            "seed": seed + i
        })
    
    # Calculate summary metrics
    success_rate = successful_episodes / num_episodes
    avg_reward = total_reward / num_episodes
    avg_achievements = total_achievements / num_episodes
    
    # Create evaluation summary
    evaluation_summary = {
        "evaluation_metadata": {
            "num_trajectories": num_episodes,
            "environment_name": "your_env",
            "timestamp": timestamp,
            "total_episodes": num_episodes,
            "successful_episodes": successful_episodes
        },
        "models_evaluated": [model_name],
        "difficulties_evaluated": [difficulty],
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_achievements": avg_achievements,
        "results": results
    }
    
    # Save evaluation summary
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"Evaluation complete!")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Results saved to: {output_dir}")
    
    return evaluation_summary

if __name__ == "__main__":
    asyncio.run(run_evaluation())
```

### Step 12: Directory Structure for Traces

Your evaluation script should create this structure:

```
src/evals/your_env/
└── run_20250704_143022/           # Timestamp format: YYYYMMDD_HHMMSS
    ├── evaluation_summary.json     # Required: Evaluation metadata
    └── traces/                     # Required: Individual trace files
        ├── trace_uuid_1.json      # One file per episode
        ├── trace_uuid_2.json
        └── ...
```

### Step 13: Trace Format Requirements

Each trace file must contain:

```json
{
  "trace": {
    "partition": [
      {
        "agent_compute_step": {
          "compute_input": [
            {
              "messages": [
                {
                  "role": "system",
                  "content": "You are solving a puzzle..."
                },
                {
                  "role": "user", 
                  "content": "Current state: ..."
                }
              ]
            }
          ],
          "compute_output": [
            {
              "messages": [
                {
                  "role": "assistant",
                  "content": "I need to analyze the current state...",
                  "tool_calls": [
                    {
                      "function": {
                        "name": "take_action",
                        "arguments": "{\"action\": \"move_up\"}"
                      }
                    }
                  ]
                }
              ]
            }
          ]
        }
      },
      {
        "environment_compute_steps": [
          {
            "compute_input": {
              "action": "move_up",
              "turn": 0
            },
            "compute_output": {
              "observation": {
                "state": "...",
                "reward_last": 0.1,
                "terminated": false
              }
            }
          }
        ]
      }
    ],
    "metadata": {
      "final_reward": 0.85,
      "success": true,
      "num_steps": 15,
      "model_name": "gpt-4o-mini",
      "difficulty": "easy",
      "seed": 42
    }
  }
}
```

**Critical Requirements:**
- Must have both `agent_compute_step` AND `environment_compute_steps`
- Agent steps must include `messages` with reasoning
- Tool calls should be included in agent messages
- Environment steps capture state changes

## Integrating with the Trace Viewer

### Step 14: Add Environment to Viewer Discovery

The trace viewer automatically discovers environments based on directory structure. Ensure your evaluation script saves to:

```
src/evals/your_env/run_*/
```

### Step 15: Create Trace Processor

Add to `src/viewer/streamlit_app.py`:

```python
def process_your_env_trace(trace_data: Dict) -> Dict[str, Any]:
    """Process your environment trace data for visualization."""
    processed_turns = []
    
    # Extract partitions (turns)
    partitions = trace_data.get('trace', {}).get('partition', [])
    
    for turn_idx, partition in enumerate(partitions):
        turn_data = {
            'turn_number': turn_idx + 1,
            'actions': [],
            'observations': [],
            'rewards': [],
            'agent_reasoning': "",
            'environment_response': {}
        }
        
        # Extract agent reasoning
        if 'agent_compute_step' in partition:
            agent_step = partition['agent_compute_step']
            if 'compute_output' in agent_step:
                messages = agent_step['compute_output'][0].get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        turn_data['agent_reasoning'] = msg.get('content', '')
                        
                        # Extract tool calls (actions)
                        tool_calls = msg.get('tool_calls', [])
                        for tool_call in tool_calls:
                            if tool_call.get('function'):
                                func = tool_call['function']
                                turn_data['actions'].append({
                                    'name': func.get('name', ''),
                                    'arguments': func.get('arguments', '{}')
                                })
        
        # Extract environment response
        if 'environment_compute_steps' in partition:
            env_steps = partition['environment_compute_steps']
            for step in env_steps:
                output = step.get('compute_output', {})
                turn_data['environment_response'] = output.get('observation', {})
                turn_data['rewards'].append(output.get('reward', 0.0))
        
        processed_turns.append(turn_data)
    
    # Extract metadata
    metadata = trace_data.get('trace', {}).get('metadata', {})
    
    return {
        'turns': processed_turns,
        'metadata': metadata,
        'model_name': metadata.get('model_name', 'unknown'),
        'difficulty': metadata.get('difficulty', 'unknown'),
        'final_reward': metadata.get('final_reward', 0.0),
        'success': metadata.get('success', False),
        'num_steps': metadata.get('num_steps', 0)
    }
```

### Step 16: Create Trace Renderer

Add to `src/viewer/streamlit_app.py`:

```python
def render_your_env_trace(processed_trace: Dict):
    """Render your environment trace visualization."""
    st.subheader("🎯 Your Environment Trace")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎮 Turns", len(processed_trace['turns']))
    with col2:
        st.metric("🏆 Success", "✅" if processed_trace['success'] else "❌")
    with col3:
        st.metric("💰 Final Reward", f"{processed_trace['final_reward']:.3f}")
    with col4:
        st.metric("📊 Steps", processed_trace['num_steps'])
    
    # Turn selector
    if processed_trace['turns']:
        selected_turn_idx = st.slider(
            "Select Turn",
            min_value=0,
            max_value=len(processed_trace['turns']) - 1,
            value=0,
            format="Turn %d"
        )
        
        # Display selected turn
        if 0 <= selected_turn_idx < len(processed_trace['turns']):
            render_your_env_turn(processed_trace['turns'][selected_turn_idx])

def render_your_env_turn(turn_data: Dict):
    """Render a detailed view of a single turn."""
    st.subheader(f"Turn {turn_data['turn_number']}")
    
    # Agent reasoning
    with st.expander("🧠 Agent Reasoning", expanded=True):
        if turn_data['agent_reasoning']:
            st.write(turn_data['agent_reasoning'])
        else:
            st.info("No agent reasoning recorded")
    
    # Actions taken
    if turn_data['actions']:
        with st.expander("🎯 Actions", expanded=True):
            for action in turn_data['actions']:
                st.code(f"{action['name']}: {action['arguments']}")
    
    # Environment response
    if turn_data['environment_response']:
        with st.expander("🌍 Environment Response", expanded=False):
            st.json(turn_data['environment_response'])
    
    # Rewards
    if turn_data['rewards']:
        with st.expander("💰 Rewards", expanded=False):
            for i, reward in enumerate(turn_data['rewards']):
                st.metric(f"Reward {i+1}", f"{reward:.3f}")
```

### Step 17: Integrate with Main Viewer Logic

Add your environment to the trace processing logic in `render_trace_visualization()`:

```python
def render_trace_visualization(trajectory: Dict, conn):
    """Render step-by-step trace visualization."""
    # ... existing code ...
    
    # Determine environment type
    env_type = "crafter"  # default
    meta_env_name = ''
    if 'environment_name' in trajectory:
        meta_env_name = str(trajectory.get('environment_name', ''))
    elif trajectory.get('metadata'):
        meta_env_name = str(trajectory['metadata'].get('environment_name', ''))
    
    # Add your environment detection
    if "your_env" in meta_env_name.lower():
        env_type = "your_env"
    elif "sokoban" in meta_env_name.lower():
        env_type = "sokoban"
    # ... other environments ...
    
    # Process trace data based on environment type
    if env_type == "your_env":
        processed_trace = process_your_env_trace(trace_data)
    elif env_type == "sokoban":
        processed_trace = process_sokoban_trace(trace_data)
    # ... other environments ...
    
    # Render based on environment type
    if env_type == "your_env":
        render_your_env_trace(processed_trace)
    elif env_type == "sokoban":
        render_sokoban_trace(processed_trace)
    # ... other environments ...
```

### Step 18: Test Viewer Integration

1. **Run your evaluation script:**
```bash
cd src/synth_env/examples/your_env
python agent_demos/run_evaluation.py
```

2. **Start the trace viewer:**
```bash
cd src/viewer
streamlit run streamlit_app.py --server.port 8501
```

3. **Sync traces in viewer:**
   - Click "🔄 Sync" button in sidebar
   - Select "your_env" environment
   - Browse your evaluation runs and traces

### Step 19: Debugging Trace Issues

Common issues and solutions:

**Problem: Traces not showing up**
- Check directory structure matches `src/evals/your_env/run_*/`
- Ensure `evaluation_summary.json` exists
- Verify trace files are valid JSON

**Problem: "No agent reasoning" warnings**
- Ensure traces have `agent_compute_step` sections
- Check that `messages` array contains agent responses
- Verify tool calls are properly formatted

**Problem: Viewer crashes on your environment**
- Add error handling in your trace processor
- Check that all expected fields exist in trace data
- Use `st.error()` for debugging instead of crashing

**Problem: Missing metadata**
- Ensure trace metadata includes required fields
- Check that environment name is properly set
- Verify model name and difficulty are recorded

### Step 20: Advanced Viewer Features

You can enhance your viewer with:

**Custom visualizations:**
```python
# Add charts, graphs, or custom displays
import plotly.express as px

def render_your_env_analytics(processed_trace: Dict):
    # Create reward progression chart
    rewards = [turn['rewards'][0] for turn in processed_trace['turns'] if turn['rewards']]
    fig = px.line(y=rewards, title="Reward Progression")
    st.plotly_chart(fig)
```

**Interactive elements:**
```python
# Add buttons, sliders, or other controls
if st.button("Show Detailed Analysis"):
    analyze_trace_patterns(processed_trace)
```

**Export functionality:**
```python
# Allow users to export processed data
if st.button("Export Trace Data"):
    st.download_button(
        "Download JSON",
        data=json.dumps(processed_trace, indent=2),
        file_name=f"trace_{trajectory['trace_id']}.json",
        mime="application/json"
    )
```

## Final Integration Checklist

- [ ] Evaluation script uses Synth SDK tracing
- [ ] Traces saved to correct directory structure
- [ ] Trace format includes agent reasoning steps
- [ ] Environment detection added to viewer
- [ ] Trace processor implemented
- [ ] Trace renderer implemented
- [ ] Integration tested end-to-end
- [ ] Error handling added for edge cases
- [ ] Documentation updated with examples

Following these steps will create a fully integrated environment with comprehensive trace visualization capabilities!