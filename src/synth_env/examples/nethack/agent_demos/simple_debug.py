#!/usr/bin/env python3
"""
Simple debug script for NetHack agent response parsing.
"""

import asyncio
import json
import logging
from synth_ai.zyk import LM
from src.synth_env.examples.nethack.agent_demos.test_synth_react import NetHackReActAgent

# Set up logging to see all details
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_agent_response():
    """Debug the agent response parsing."""
    
    print("🔍 Starting agent response debug...")
    
    # Create agent
    llm = LM(model_name="gemini-1.5-flash-latest", formatting_model_name="gemini-1.5-flash-latest", temperature=0.0)
    agent = NetHackReActAgent(llm, max_turns=5)
    
    # Set a simple system prompt
    agent.system_prompt = """You are a NetHack player. Use the tools available to play the game."""
    
    # Simple observation
    obs = """
=== NetHack Observation ===
Message: Welcome to NetHack! You are a knight.
Map: You see a room with a monster 'u' and some gold '$'.

You need to decide your next action. You can move around or interact with the environment.
"""
    
    print("📝 Observation:")
    print(obs)
    
    print("\n🤖 Calling agent...")
    
    try:
        # Get LLM response directly
        response = await agent.llm.respond_async(
            system_message=agent.system_prompt,
            user_message=obs,
            tools=agent.tools
        )
        
        print(f"\n📋 Raw LLM Response:")
        print(f"  Type: {type(response)}")
        print(f"  Dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Check specific attributes
        for attr in ['tool_calls', 'choices', 'message', 'content']:
            if hasattr(response, attr):
                value = getattr(response, attr)
                print(f"  {attr}: {type(value)} = {value}")
        
        # Check if it has tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
            print(f"\n🔧 Tool calls found: {len(tool_calls)}")
            
            for i, tool_call in enumerate(tool_calls):
                print(f"\n  Tool call {i+1}:")
                print(f"    Type: {type(tool_call)}")
                print(f"    Dir: {[attr for attr in dir(tool_call) if not attr.startswith('_')]}")
                
                # Check for function attribute
                if hasattr(tool_call, 'function'):
                    func = tool_call.function
                    print(f"    Function: {type(func)}")
                    print(f"    Function dir: {[attr for attr in dir(func) if not attr.startswith('_')]}")
                    
                    # Check for name and arguments
                    if hasattr(func, 'name'):
                        print(f"    Function name: {func.name}")
                    else:
                        print(f"    ❌ Function has no 'name' attribute!")
                    
                    if hasattr(func, 'arguments'):
                        print(f"    Function arguments: {func.arguments}")
                    else:
                        print(f"    ❌ Function has no 'arguments' attribute!")
        
        # Try calling the agent's decide method
        print(f"\n🎯 Calling agent.decide()...")
        decision = await agent.decide(obs)
        print(f"✅ Decision: {decision}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_agent_response()) 