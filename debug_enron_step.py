#!/usr/bin/env python3

import asyncio
import httpx

async def debug_enron_step():
    """Debug the Enron step issue by examining the error more carefully"""
    
    async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        try:
            # 1. Initialize Enron environment  
            print("1. Initializing Enron environment...")
            initial_state = {
                "question": "What was the main topic discussed in emails about project Alpha?",
                "answer": "Project timeline and budget concerns",
                "emails": []  
            }
            
            resp = await client.post(
                "/env/Enron/initialize",
                json={
                    "initial_state": initial_state,
                    "config": {}
                }
            )
            print(f"   Initialize status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"   Initialize failed: {resp.text}")
                return
                
            init_data = resp.json()
            instance_id = init_data["env_id"]
            print(f"   ✅ Created instance: {instance_id}")
            print(f"   Observation: {init_data['observation'][:200]}...")  # First 200 chars
            
            # 2. Try step with different args to see what works
            test_cases = [
                {
                    "name": "Simple search",
                    "action": {
                        "tool_calls": [{"tool": "search_emails", "args": {
                            "inbox": "test@example.com",
                            "keywords": ["project"]
                        }}]
                    }
                },
                {
                    "name": "Search with all args",
                    "action": {
                        "tool_calls": [{"tool": "search_emails", "args": {
                            "inbox": "test@example.com", 
                            "keywords": ["project", "alpha"],
                            "from_addr": None,
                            "to_addr": None,
                            "sent_after": None,
                            "sent_before": None,
                            "max_results": 5
                        }}]
                    }
                },
                {
                    "name": "Answer question",
                    "action": {
                        "tool_calls": [{"tool": "answer_question", "args": {
                            "answer": "Testing answer"
                        }}]
                    }
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                print(f"\n2.{i+1}. Testing: {test_case['name']}")
                
                step_resp = await client.post(
                    "/env/Enron/step",
                    json={
                        "env_id": instance_id,
                        "request_id": f"debug_step_{i}",
                        "action": test_case["action"]
                    }
                )
                print(f"   Step status: {step_resp.status_code}")
                print(f"   Response: {step_resp.text}")
                
                if step_resp.status_code == 200:
                    print("   ✅ Success!")
                    break  # If this works, we can stop here
                else:
                    print(f"   ❌ Failed: {step_resp.text}")
            
            # 3. Terminate
            print("\n3. Terminating...")
            term_resp = await client.post(
                "/env/Enron/terminate",
                json={"env_id": instance_id}
            )
            print(f"   Terminate status: {term_resp.status_code}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_enron_step()) 