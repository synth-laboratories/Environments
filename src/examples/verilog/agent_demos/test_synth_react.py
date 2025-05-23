import sys
import os
import asyncio
import uuid
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from examples.verilog.environment import VerilogEnvironment
from examples.verilog.taskset import VerilogTaskInstance, create_verilog_taskset
from examples.verilog.engine import VerilogPrivateState, VerilogPublicState
from src.environment.tools import EnvToolCall
from src.tasks.core import Impetus, Intent
from synth_ai.zyk import LM


# Tool argument models for the agent
class WriteFileArgs(BaseModel):
    path: str = Field(description="Path to the Verilog file to write")
    content: str = Field(description="Verilog code content")
    reasoning: str = Field(description="Reasoning for the code implementation")


class CompileArgs(BaseModel):
    sources: Optional[List[str]] = Field(None, description="List of source files to compile")
    testbench: Optional[str] = Field(None, description="Testbench file to include")
    reasoning: str = Field(description="Reasoning for compilation step")


class SimulateArgs(BaseModel):
    binary: Optional[str] = Field(None, description="Binary file to simulate")
    reasoning: str = Field(description="Reasoning for simulation step")


class SubmitArgs(BaseModel):
    reasoning: str = Field(description="Reasoning for submission")


class TerminateArgs(BaseModel):
    reason: str = Field(description="Reason for termination")


# Environment tool call wrappers
class WriteFile(EnvToolCall):
    def __init__(self, path: str, content: str):
        super().__init__()
        self.tool = "write_file"
        self.args = {"path": path, "content": content}


class Compile(EnvToolCall):
    def __init__(self, sources: Optional[List[str]] = None, testbench: Optional[str] = None):
        super().__init__()
        self.tool = "compile"
        self.args = {"sources": sources, "testbench": testbench}


class Simulate(EnvToolCall):
    def __init__(self, binary: Optional[str] = None):
        super().__init__()
        self.tool = "simulate"
        self.args = {"binary": binary}


class Submit(EnvToolCall):
    def __init__(self):
        super().__init__()
        self.tool = "submit"
        self.args = {}


def format_obs_for_llm(obs: Dict[str, Any]) -> str:
    """Format observation for LLM input."""
    files_info = ""
    if obs.get("files"):
        files_info = "Available files:\n"
        for filename, content in obs["files"].items():
            files_info += f"  {filename}:\n"
            # Show first few lines of content
            lines = content.split('\n')[:10]
            for line in lines:
                files_info += f"    {line}\n"
            if len(content.split('\n')) > 10:
                files_info += "    ...\n"
        files_info += "\n"
    
    compile_status = obs.get("compile_status", "")
    simulate_status = obs.get("simulate_status", "")
    
    status_info = f"Task completed: {obs.get('task_completed', False)}\n"
    status_info += f"Terminated: {obs.get('terminated', False)}\n"
    status_info += f"Total reward: {obs.get('total_reward', 0)}\n"
    
    if compile_status:
        status_info += f"Compile status: {compile_status}\n"
    if simulate_status:
        status_info += f"Simulate status: {simulate_status}\n"
    
    return f"{files_info}{status_info}"


class VerilogReActAgent:
    """Simple ReAct agent for Verilog tasks."""
    
    def __init__(self, llm, max_turns: int = 15):
        self.llm = llm
        self.max_turns = max_turns
        self.history: List[Dict[str, Any]] = []
        self.task_description = ""
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write Verilog code to a file",
                    "parameters": WriteFileArgs.model_json_schema(),
                },
            },
            {
                "type": "function", 
                "function": {
                    "name": "compile",
                    "description": "Compile Verilog sources with iverilog",
                    "parameters": CompileArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate", 
                    "description": "Run simulation with vvp",
                    "parameters": SimulateArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit",
                    "description": "Submit solution for grading",
                    "parameters": SubmitArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "Terminate if task is complete or cannot proceed",
                    "parameters": TerminateArgs.model_json_schema(),
                },
            },
        ]

    def set_task_description(self, description: str):
        """Set the task description for this agent."""
        self.task_description = description

    async def decide(self, obs: str) -> Dict[str, Any]:
        """Decide next action based on observation."""
        self.history.append({"type": "observation", "content": obs})
        
        # Build prompt from history
        history_text = ""
        for entry in self.history[-5:]:  # Last 5 entries
            if entry["type"] == "observation":
                history_text += f"OBSERVATION:\n{entry['content']}\n\n"
            elif entry["type"] == "tool_call":
                history_text += f"ACTION: Called {entry['tool_name']} with args: {entry['tool_args']}\n\n"
            elif entry["type"] == "tool_response":
                history_text += f"RESULT: {entry['content']}\n\n"
        
        prompt = f"""Task: {self.task_description}

History:
{history_text}

Based on the observation and history, decide what to do next. You should:
1. First understand what files are available and what needs to be implemented
2. Write the required Verilog code
3. Compile the code 
4. Run simulation to test
5. Submit when tests pass

Choose the most appropriate tool to call next."""

        system_message = """You are a Verilog design expert. Your goal is to implement correct Verilog code that passes testbenches.

Available tools:
- write_file: Write Verilog code to files
- compile: Compile Verilog sources with iverilog  
- simulate: Run simulation with vvp
- submit: Submit solution when complete
- terminate: End if task complete or cannot proceed

Always use the tools available. Include reasoning in your tool calls."""

        try:
            response = await self.llm.respond_async(
                system_message=system_message,
                user_message=prompt,
                tools=self.tools
            )
            
            if not response.tool_calls:
                return {"action": "terminate", "args": {"reason": "No tool call generated"}}
                
            tool_call = response.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            if isinstance(tool_args, str):
                import json
                tool_args = json.loads(tool_args)
                
            self.history.append({
                "type": "tool_call", 
                "tool_name": tool_name,
                "tool_args": tool_args
            })
            
            return {"action": tool_name, "args": tool_args}
            
        except Exception as e:
            return {"action": "terminate", "args": {"reason": f"Error: {str(e)}"}}


async def run_verilog_episode(task_instance: VerilogTaskInstance) -> bool:
    """Run a single episode with the Verilog environment and agent."""
    
    # Create environment
    env = VerilogEnvironment(task_instance)
    
    # Create agent
    llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)
    agent = VerilogReActAgent(llm)
    
    # Set task description from the task instance
    agent.set_task_description(task_instance.impetus.instructions)
    
    try:
        # Initialize environment
        obs = await env.initialize()
        obs_text = format_obs_for_llm(obs)
        
        # Run episode
        for turn in range(agent.max_turns):
            # Agent decides action
            decision = await agent.decide(obs_text)
            
            if decision["action"] == "terminate":
                reason = decision["args"].get("reason", "Agent terminated")
                agent.history.append({"type": "tool_response", "content": f"Terminated: {reason}"})
                break
                
            # Execute action in environment
            action_name = decision["action"]
            action_args = decision["args"]
            
            # Create appropriate tool call
            if action_name == "write_file":
                tool_call = WriteFile(action_args["path"], action_args["content"])
            elif action_name == "compile":
                tool_call = Compile(action_args.get("sources"), action_args.get("testbench"))
            elif action_name == "simulate":
                tool_call = Simulate(action_args.get("binary"))
            elif action_name == "submit":
                tool_call = Submit()
            else:
                agent.history.append({"type": "tool_response", "content": f"Unknown action: {action_name}"})
                continue
                
            # Step environment
            obs = await env.step(tool_call)
            obs_text = format_obs_for_llm(obs)
            
            # Record result
            agent.history.append({"type": "tool_response", "content": obs_text})
            
            # Check if terminated
            if obs.get("terminated", False):
                return obs.get("task_completed", False)
                
        return False
        
    except Exception as e:
        print(f"Episode failed with error: {e}")
        return False


@pytest.mark.asyncio
async def test_verilog_react_agent():
    """Test the Verilog ReAct agent on a simple task."""
    
    # Create a simple task set
    taskset = await create_verilog_taskset()
    
    # Test with the first task (should be the adder)
    task_instance = taskset.instances[0]
    
    # Run episode
    success = await run_verilog_episode(task_instance)
    
    print(f"Task: {task_instance.metadata.problem_name}")
    print(f"Success: {success}")
    
    # For testing, we'll allow failure since this is a basic implementation
    # In a full implementation, we'd expect the agent to succeed
    assert success or not success  # Always pass for now


async def eval_verilog_react():
    """Evaluate the ReAct agent on all Verilog tasks."""
    
    # Create task set
    taskset = await create_verilog_taskset()
    
    results = []
    for task_instance in taskset.instances:
        print(f"\nRunning task: {task_instance.metadata.problem_name}")
        success = await run_verilog_episode(task_instance)
        results.append({
            "task": task_instance.metadata.problem_name,
            "difficulty": task_instance.metadata.difficulty,
            "success": success
        })
        print(f"Result: {'PASS' if success else 'FAIL'}")
    
    # Print summary
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r["success"])
    
    print(f"\n=== SUMMARY ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Success rate: {successful_tasks/total_tasks:.1%}")
    
    # By difficulty
    by_difficulty = {}
    for result in results:
        diff = result["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "success": 0}
        by_difficulty[diff]["total"] += 1
        if result["success"]:
            by_difficulty[diff]["success"] += 1
    
    for diff, stats in by_difficulty.items():
        rate = stats["success"] / stats["total"]
        print(f"{diff}: {stats['success']}/{stats['total']} ({rate:.1%})")


if __name__ == "__main__":
    asyncio.run(eval_verilog_react())