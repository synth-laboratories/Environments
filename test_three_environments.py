#!/usr/bin/env python3

import asyncio
import json
import httpx
from typing import Dict, Any

async def test_environment_health():
    """Test health endpoint and verify all environments are registered"""
    print("=== TESTING SERVICE HEALTH ===")
    
    async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        try:
            health_resp = await client.get("/health")
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                print(f"✅ Service healthy")
                print(f"   Supported environments: {health_data['supported_environments']}")
                return health_data['supported_environments']
            else:
                print(f"❌ Health check failed: {health_resp.status_code}")
                return []
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return []

async def test_verilog_environment():
    """Test Verilog environment through both APIs"""
    print("\n=== TESTING VERILOG ENVIRONMENT ===")
    
    async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        try:
            # Test new API - initialize with minimal config
            print("1. Testing Verilog with new API...")
            resp = await client.post(
                "/env/Verilog/initialize",
                json={
                    "initial_state": {},  # Empty initial state
                    "config": {}
                }
            )
            print(f"   Initialize response status: {resp.status_code}")
            
            if resp.status_code == 200:
                init_data = resp.json()
                instance_id = init_data["env_id"]
                observation = init_data["observation"]
                
                print(f"   ✅ Created Verilog instance: {instance_id}")
                print(f"   ✅ Observation type: {type(observation)}")
                print(f"   ✅ Observation keys: {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}")
                
                # Try a simple step
                step_resp = await client.post(
                    "/env/Verilog/step",
                    json={
                        "env_id": instance_id,
                        "request_id": "verilog_test_step",
                        "action": {
                            "tool_calls": [{"tool": "write_file", "args": {"filename": "test.v", "content": "// test comment"}}]
                        }
                    }
                )
                print(f"   Step response status: {step_resp.status_code}")
                
                if step_resp.status_code == 200:
                    step_data = step_resp.json()
                    step_obs = step_data.get("observation", {})
                    print(f"   ✅ Step successful - observation keys: {list(step_obs.keys()) if isinstance(step_obs, dict) else 'Not a dict'}")
                else:
                    print(f"   ⚠️ Step failed: {step_resp.text}")
                
                # Terminate
                term_resp = await client.post(
                    "/env/Verilog/terminate",
                    json={"env_id": instance_id}
                )
                print(f"   ✅ Terminated: {term_resp.status_code == 200}")
                
            else:
                print(f"   ❌ Initialize failed: {resp.text}")
                
        except Exception as e:
            print(f"   ❌ Verilog test failed: {e}")

async def test_enron_environment():
    """Test Enron environment through both APIs"""
    print("\n=== TESTING ENRON ENVIRONMENT ===")
    
    async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        try:
            # Test new API - need to provide a task instance for Enron
            print("1. Testing Enron with new API...")
            
            # Enron requires a task instance - let's try with a minimal one
            initial_state = {
                "question": "What was the main topic discussed in emails about project Alpha?",
                "answer": "Project timeline and budget concerns",
                "emails": []  # Minimal email dataset
            }
            
            resp = await client.post(
                "/env/Enron/initialize",
                json={
                    "initial_state": initial_state,
                    "config": {}
                }
            )
            print(f"   Initialize response status: {resp.status_code}")
            
            if resp.status_code == 200:
                init_data = resp.json()
                instance_id = init_data["env_id"]
                observation = init_data["observation"]
                
                print(f"   ✅ Created Enron instance: {instance_id}")
                print(f"   ✅ Observation type: {type(observation)}")
                print(f"   ✅ Observation keys: {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}")
                
                # Try a search action
                step_resp = await client.post(
                    "/env/Enron/step",
                    json={
                        "env_id": instance_id,
                        "request_id": "enron_test_step",
                        "action": {
                            "tool_calls": [{"tool": "search_emails", "args": {
                                "inbox": "test@example.com",
                                "keywords": ["project", "alpha"],
                                "max_results": 5
                            }}]
                        }
                    }
                )
                print(f"   Step response status: {step_resp.status_code}")
                
                if step_resp.status_code == 200:
                    step_data = step_resp.json()
                    step_obs = step_data.get("observation", {})
                    print(f"   ✅ Step successful - observation keys: {list(step_obs.keys()) if isinstance(step_obs, dict) else 'Not a dict'}")
                else:
                    print(f"   ⚠️ Step failed: {step_resp.text}")
                
                # Terminate
                term_resp = await client.post(
                    "/env/Enron/terminate",
                    json={"env_id": instance_id}
                )
                print(f"   ✅ Terminated: {term_resp.status_code == 200}")
                
            else:
                print(f"   ❌ Initialize failed: {resp.text}")
                
        except Exception as e:
            print(f"   ❌ Enron test failed: {e}")

async def test_nethack_environment():
    """Test NetHack environment through both APIs"""
    print("\n=== TESTING NETHACK ENVIRONMENT ===")
    
    async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        try:
            # Test new API
            print("1. Testing NetHack with new API...")
            resp = await client.post(
                "/env/NetHack/initialize",
                json={
                    "initial_state": {},  # Empty initial state
                    "config": {}
                }
            )
            print(f"   Initialize response status: {resp.status_code}")
            
            if resp.status_code == 200:
                init_data = resp.json()
                instance_id = init_data["env_id"]
                observation = init_data["observation"]
                
                print(f"   ✅ Created NetHack instance: {instance_id}")
                print(f"   ✅ Observation type: {type(observation)}")
                print(f"   ✅ Observation keys: {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}")
                
                # Try a movement action
                step_resp = await client.post(
                    "/env/NetHack/step",
                    json={
                        "env_id": instance_id,
                        "request_id": "nethack_test_step",
                        "action": {
                            "tool_calls": [{"tool": "move", "args": {"direction": "north"}}]
                        }
                    }
                )
                print(f"   Step response status: {step_resp.status_code}")
                
                if step_resp.status_code == 200:
                    step_data = step_resp.json()
                    step_obs = step_data.get("observation", {})
                    print(f"   ✅ Step successful - observation keys: {list(step_obs.keys()) if isinstance(step_obs, dict) else 'Not a dict'}")
                else:
                    print(f"   ⚠️ Step failed: {step_resp.text}")
                
                # Terminate
                term_resp = await client.post(
                    "/env/NetHack/terminate",
                    json={"env_id": instance_id}
                )
                print(f"   ✅ Terminated: {term_resp.status_code == 200}")
                
            else:
                print(f"   ❌ Initialize failed: {resp.text}")
                
        except Exception as e:
            print(f"   ❌ NetHack test failed: {e}")

async def main():
    """Run all environment tests"""
    print("🚀 Starting comprehensive environment testing...")
    
    # Wait for service to start
    await asyncio.sleep(3)
    
    # Test health first
    supported_envs = await test_environment_health()
    
    if "Verilog" in supported_envs:
        await test_verilog_environment()
    else:
        print("\n❌ Verilog not found in supported environments")
    
    if "Enron" in supported_envs:
        await test_enron_environment()
    else:
        print("\n❌ Enron not found in supported environments")
    
    if "NetHack" in supported_envs:
        await test_nethack_environment()
    else:
        print("\n❌ NetHack not found in supported environments")
    
    print("\n🏁 Testing complete!")

if __name__ == "__main__":
    asyncio.run(main()) 