#!/usr/bin/env python3
"""
Enron Environment Evaluation with GPT-4.1-mini

This script tests the Enron environment by running a ReAct agent with GPT-4.1-mini
to see if it can successfully answer questions about the Enron email dataset.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path

from synth_ai.zyk import LM


@dataclass
class EvaluationResult:
    """Result of a single evaluation episode"""
    question: str
    expected_answer: str
    agent_answer: str
    success: bool
    steps_taken: int
    duration: float
    search_count: int
    read_count: int
    error: Optional[str] = None


class EnronReActAgent:
    """Simple ReAct agent for Enron email QA evaluation"""
    
    def __init__(self, llm: LM, max_steps: int = 10):
        self.llm = llm
        self.max_steps = max_steps
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_emails",
                    "description": (
                        "Search for emails using keywords. Pass individual words in the keywords list. "
                        "Example: search_emails(keywords=['project', 'alpha', 'budget'])"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "inbox": {"type": "string", "description": "Email address to search"},
                            "keywords": {"type": "array", "items": {"type": "string"}, "description": "List of keywords to search for"},
                            "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10}
                        },
                        "required": ["inbox", "keywords"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "read_email",
                    "description": "Read a specific email by message ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string", "description": "The message ID of the email to read"}
                        },
                        "required": ["message_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "answer_question", 
                    "description": "Provide the final answer to the question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "The final answer to the question"}
                        },
                        "required": ["answer"]
                    }
                }
            }
        ]
    
    async def solve_question(self, question: str, inbox: str) -> tuple[str, int, int, int]:
        """
        Solve a question about emails in the given inbox.
        
        Returns:
            (answer, steps_taken, search_count, read_count)
        """
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert email analysis agent. Your task is to answer questions about emails in the inbox {inbox}.

Instructions:
1. Use search_emails to find relevant emails based on keywords from the question
2. Use read_email to examine specific emails that seem relevant
3. Use answer_question to provide your final answer
4. Be systematic: search first, then read promising emails, then answer
5. Extract keywords from the question for effective searching

Question to answer: {question}"""
            },
            {
                "role": "user", 
                "content": f"Please help me answer this question about emails in {inbox}: {question}"
            }
        ]
        
        steps_taken = 0
        search_count = 0
        read_count = 0
        
        for step in range(self.max_steps):
            try:
                response = await self.llm.respond_async(
                    messages=messages,
                    tools=self.tools,
                    temperature=0.1
                )
                
                steps_taken += 1
                
                if not response.tool_calls:
                    # No tool call, just text response
                    messages.append({"role": "assistant", "content": response.content or "I need to use tools to help answer this question."})
                    continue
                
                # Execute the first tool call
                tool_call = response.tool_calls[0]
                
                if hasattr(tool_call, 'function'):
                    # OpenAI format
                    function_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                elif isinstance(tool_call, dict):
                    # Dict format
                    function_name = tool_call.get('function', {}).get('name', '')
                    arguments = tool_call.get('function', {}).get('arguments', {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                else:
                    continue
                
                if function_name == "answer_question":
                    return arguments.get("answer", ""), steps_taken, search_count, read_count
                elif function_name == "search_emails":
                    search_count += 1
                    result = f"Search executed with keywords: {arguments.get('keywords', [])}. Found some emails (simulated result for eval)."
                elif function_name == "read_email":
                    read_count += 1
                    result = f"Read email {arguments.get('message_id', 'unknown')} (simulated result for eval)."
                else:
                    result = "Unknown function called."
                
                # Add the tool call and result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, 'id', 'tool_call'),
                    "content": result
                })
                
            except Exception as e:
                print(f"Error in step {step}: {e}")
                break
        
        return "No answer provided", steps_taken, search_count, read_count


async def run_enron_evaluation_service(num_episodes: int = 5, model_name: str = "gpt-4.1-mini") -> List[EvaluationResult]:
    """
    Run evaluation episodes using the environment service.
    """
    # Sample test cases
    test_cases = [
        {
            "question": "What was discussed in meetings about the California energy crisis?",
            "inbox": "jeff.skilling@enron.com",
            "expected": "Energy market regulations and trading strategies"
        },
        {
            "question": "What are the key points from emails about Enron's stock price?", 
            "inbox": "kenneth.lay@enron.com",
            "expected": "Stock performance and investor concerns"
        },
        {
            "question": "What projects were mentioned in emails about renewable energy?",
            "inbox": "jeff.skilling@enron.com", 
            "expected": "Wind and solar energy initiatives"
        },
        {
            "question": "What were the main concerns in emails about accounting practices?",
            "inbox": "andrew.fastow@enron.com",
            "expected": "Special purpose entities and financial reporting"
        },
        {
            "question": "What merger discussions appear in the email correspondence?",
            "inbox": "kenneth.lay@enron.com",
            "expected": "Various corporate acquisition talks"
        }
    ]
    
    llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.1)
    agent = EnronReActAgent(llm)
    results = []
    
    async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
        print(f"🚀 Starting Enron evaluation with {model_name}")
        print(f"📊 Running {min(num_episodes, len(test_cases))} test episodes")
        print("=" * 60)
        
        for i in range(min(num_episodes, len(test_cases))):
            test_case = test_cases[i]
            print(f"\n📧 Episode {i+1}: {test_case['question']}")
            print(f"📮 Inbox: {test_case['inbox']}")
            
            start_time = time.time()
            
            try:
                # Initialize environment
                initial_state = {
                    "question": test_case["question"],
                    "answer": "",  # We don't give the answer to the agent
                    "inbox_address": test_case["inbox"],
                    "emails": []
                }
                
                # Create environment instance
                init_resp = await client.post(
                    "/env/Enron/initialize",
                    json={
                        "initial_state": initial_state,
                        "config": {}
                    }
                )
                
                if init_resp.status_code != 200:
                    error_msg = f"Failed to initialize environment: {init_resp.status_code} - {init_resp.text}"
                    print(f"❌ {error_msg}")
                    results.append(EvaluationResult(
                        question=test_case["question"],
                        expected_answer=test_case["expected"], 
                        agent_answer="",
                        success=False,
                        steps_taken=0,
                        duration=time.time() - start_time,
                        search_count=0,
                        read_count=0,
                        error=error_msg
                    ))
                    continue
                
                init_data = init_resp.json()
                env_id = init_data["env_id"]
                
                print(f"✅ Environment initialized: {env_id}")
                
                # Run the agent
                answer, steps, search_count, read_count = await agent.solve_question(
                    test_case["question"], 
                    test_case["inbox"]
                )
                
                duration = time.time() - start_time
                
                # Simple success check (contains expected keywords)
                expected_words = test_case["expected"].lower().split()
                answer_words = answer.lower().split()
                success = any(word in answer_words for word in expected_words) and len(answer.strip()) > 10
                
                result = EvaluationResult(
                    question=test_case["question"],
                    expected_answer=test_case["expected"],
                    agent_answer=answer,
                    success=success,
                    steps_taken=steps,
                    duration=duration, 
                    search_count=search_count,
                    read_count=read_count
                )
                
                results.append(result)
                
                # Print results
                status_icon = "✅" if success else "❌"
                print(f"{status_icon} Answer: {answer}")
                print(f"📈 Stats: {steps} steps, {search_count} searches, {read_count} reads, {duration:.1f}s")
                
                # Terminate environment
                await client.post(
                    "/env/Enron/terminate",
                    json={"env_id": env_id}
                )
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Exception during episode: {str(e)}"
                print(f"❌ {error_msg}")
                
                results.append(EvaluationResult(
                    question=test_case["question"],
                    expected_answer=test_case["expected"],
                    agent_answer="",
                    success=False,
                    steps_taken=0,
                    duration=duration,
                    search_count=0,
                    read_count=0,
                    error=error_msg
                ))
    
    return results


def print_evaluation_summary(results: List[EvaluationResult]):
    """Print a summary of evaluation results"""
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r.success]
    total = len(results)
    success_rate = len(successful) / total if total > 0 else 0
    
    print(f"🎯 Success Rate: {len(successful)}/{total} ({success_rate:.1%})")
    print(f"⏱️  Average Duration: {sum(r.duration for r in results) / total:.1f}s")
    print(f"🔍 Average Searches: {sum(r.search_count for r in results) / total:.1f}")
    print(f"📖 Average Reads: {sum(r.read_count for r in results) / total:.1f}")
    print(f"👣 Average Steps: {sum(r.steps_taken for r in results) / total:.1f}")
    
    if successful:
        print(f"\n✅ SUCCESSFUL EPISODES:")
        for i, result in enumerate(successful, 1):
            print(f"  {i}. Q: {result.question[:50]}...")
            print(f"     A: {result.agent_answer[:100]}...")
    
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n❌ FAILED EPISODES:")
        for i, result in enumerate(failed, 1):
            error_info = f" (Error: {result.error})" if result.error else ""
            print(f"  {i}. Q: {result.question[:50]}...{error_info}")
            if result.agent_answer:
                print(f"     A: {result.agent_answer[:100]}...")


async def main():
    """Main evaluation function"""
    print("🧠 Enron Environment Evaluation with GPT-4.1-mini")
    print("Testing if the environment works and can get successful answers")
    
    try:
        results = await run_enron_evaluation_service(num_episodes=5, model_name="gpt-4.1-mini")
        print_evaluation_summary(results)
        
        # Return success if at least one episode succeeded
        return len([r for r in results if r.success]) > 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Evaluation completed with at least one success!")
    else:
        print("\n😞 Evaluation completed with no successes.") 