# environment.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from synth_env.environment.tools import (
    EnvToolCall,
    ToolResult,
    TOOL_REGISTRY,
    register_tool,
    AbstractTool,
)
from synth_env.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_env.stateful.core import StatefulEnvironment
from synth_env.examples.enron.engine import (
    EnronEngine,
    ACTION_SEARCH,
    ACTION_READ,
    ACTION_ANSWER,
)
from synth_env.examples.enron.taskset import EnronTaskInstance


# -------- pydantic schemas (used by agent / LLM function calls)
class SearchEmailsArgs(BaseModel):
    inbox: str = Field(
        ..., description="Email address performing the search (used by tool logic)"
    )
    keywords: List[str] = Field(..., description="Keywords to AND-search for")
    from_addr: Optional[str] = None
    to_addr: Optional[str] = None
    sent_after: Optional[str] = None
    sent_before: Optional[str] = None
    max_results: int = Field(10, le=10)


class ReadEmailArgs(BaseModel):
    message_id: str


class AnswerQuestionArgs(BaseModel):
    answer: str


# --------------------------------------------------------------------------- tool implementations
class SearchEmailsTool(AbstractTool):
    name = "search_emails"
    description = "Search for emails using keywords and filters"
    call_schema = SearchEmailsArgs
    result_schema = ToolResult
    
    def __init__(self, engine: EnronEngine):
        self.engine = engine
    
    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Execute the search action
            result = await self.engine.search_emails_action(call.args)
            return ToolResult(ok=True, payload={"search_results": result})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class ReadEmailTool(AbstractTool):
    name = "read_email"
    description = "Read an email by message ID"
    call_schema = ReadEmailArgs
    result_schema = ToolResult
    
    def __init__(self, engine: EnronEngine):
        self.engine = engine
    
    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Execute the read action
            result = await self.engine.read_email_action(call.args["message_id"])
            return ToolResult(ok=True, payload={"email": result})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class AnswerQuestionTool(AbstractTool):
    name = "answer_question"
    description = "Answer the question with given answer"
    call_schema = AnswerQuestionArgs
    result_schema = ToolResult
    
    def __init__(self, engine: EnronEngine):
        self.engine = engine
    
    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Execute the answer action
            await self.engine.answer_question_action(call.args["answer"])
            return ToolResult(ok=True, payload={"answered": True, "answer": call.args["answer"]})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class TerminateTool(AbstractTool):
    name = "terminate"
    description = "Terminate the session"
    call_schema = None  # No arguments needed
    result_schema = ToolResult
    
    def __init__(self, engine: EnronEngine):
        self.engine = engine
    
    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            # Execute terminate as empty answer
            await self.engine.answer_question_action("")
            return ToolResult(ok=True, payload={"terminated": True, "answer": ""})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


# -------- observation callable (optional for formatted observations)
class SynthEnronObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: Dict[str, Any], priv: Dict[str, Any]
    ) -> InternalObservation:
        """Format observation as a comprehensive dict."""
        # Return a comprehensive dict with all relevant information
        return {
            **pub,  # Include all public state
            **priv,  # Include all private state
            "question": pub.get("question"),
            "tools": pub.get("tools", []),
            "already_answered": pub.get("already_answered", False),
            "search_results": pub.get("search_results", []),
            "search_results_count": len(pub.get("search_results", [])),
            "email": pub.get("email"),
            "email_loaded": pub.get("email") is not None,
            "tool_error": pub.get("tool_error"),
            "reward_last": priv.get("reward_last", 0),
            "total_reward": priv.get("total_reward", 0),
            "terminated": priv.get("terminated", False),
            "truncated": priv.get("truncated", False),
            "gold_answer": priv.get("gold_answer"),
        }


# --------------------------------------------------------------------------- environment
class EnronEnvironment(StatefulEnvironment):
    def __init__(
        self,
        task_instance: EnronTaskInstance,
        custom_obs: Optional[GetObservationCallable] = None,
    ):
        self.engine = EnronEngine(task_instance)
        self.custom_obs = custom_obs or SynthEnronObservationCallable()
        self.name = "Enron-QA-Env"

        # Store tool instances on self for reliable access
        self._tools_instances = {
            "search_emails": SearchEmailsTool(self.engine),
            "read_email": ReadEmailTool(self.engine),
            "answer_question": AnswerQuestionTool(self.engine),
            "terminate": TerminateTool(self.engine),
        }
        for tool_name, tool_instance in self._tools_instances.items():
            if tool_name not in TOOL_REGISTRY:
                register_tool(tool_instance)
            elif getattr(TOOL_REGISTRY[tool_name], 'engine', None) is not self.engine:
                register_tool(tool_instance)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._obs(priv, pub)

    def validate_tool_calls(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> List[EnvToolCall]:
        """Normalize and validate tool calls to EnvToolCall objects."""
        # Normalize to list format
        if isinstance(tool_calls, EnvToolCall):
            return [tool_calls]
        elif isinstance(tool_calls, dict):
            # Single tool call as dict
            tool_name = tool_calls.get("tool")
            tool_args = tool_calls.get("args", {})
            if tool_name not in ["search_emails", "read_email", "answer_question", "terminate"]:
                raise ValueError(f"Unknown tool: {tool_name}. Expected one of: search_emails, read_email, answer_question, terminate")
            return [EnvToolCall(tool=tool_name, args=tool_args)]
        elif isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Received empty list of tool calls.")
            
            # Handle nested list format
            if isinstance(tool_calls[0], list):
                if not tool_calls[0]:
                    raise ValueError("Received empty inner list of tool calls.")
                tool_calls = tool_calls[0]  # Flatten one level
            
            # Convert list of dicts to EnvToolCall objects
            result = []
            for call_data in tool_calls:
                if isinstance(call_data, EnvToolCall):
                    result.append(call_data)
                elif isinstance(call_data, dict):
                    tool_name = call_data.get("tool")
                    tool_args = call_data.get("args", {})
                    if tool_name not in ["search_emails", "read_email", "answer_question", "terminate"]:
                        raise ValueError(f"Unknown tool: {tool_name}. Expected one of: search_emails, read_email, answer_question, terminate")
                    result.append(EnvToolCall(tool=tool_name, args=tool_args))
                else:
                    raise TypeError(f"Unexpected type in tool_calls: {type(call_data)}")
            return result
        else:
            raise TypeError(f"Unexpected type for tool_calls: {type(tool_calls)}")

    async def step(
        self,
        calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]], List[Dict[str, Any]], Dict[str, Any]],
    ) -> InternalObservation:
        # Validate and normalize tool calls
        validated_calls = self.validate_tool_calls(calls)
        
        if not validated_calls:
            raise ValueError("No valid tool calls provided")
        
        # Use the first tool call (Enron handles one tool at a time)
        tool_call = validated_calls[0]
        tool_name = tool_call.tool
        tool_to_execute = self._tools_instances.get(tool_name)

        if not tool_to_execute:
            tool_to_execute = TOOL_REGISTRY.get(tool_name)
            if not tool_to_execute:
                raise ValueError(f"Tool '{tool_name}' not found.")

        tool_result: ToolResult = await tool_to_execute(tool_call)

        public_payload_for_engine = (
            tool_result.payload if tool_result.ok and tool_result.payload else {}
        )
        if not tool_result.ok:
            public_payload_for_engine["tool_error"] = tool_result.error

        priv, pub = await self.engine._step_engine(public_payload_for_engine)
        return await self._obs(priv, pub)

    async def terminate(self) -> InternalObservation:
        self.engine.close_db()
        priv_state_on_terminate = {
            "reward_last": 0,
            "total_reward": self.engine.total_reward,
            "terminated": True,
            "truncated": False,
            "gold_answer": self.engine._sample()["answer"],
        }
        pub_state_on_terminate = {
            "question": self.engine._sample()["question"],
            "tools": [],
            "already_answered": self.engine.answered,
            "status": "terminated_by_env",
        }
        return await self._obs(priv_state_on_terminate, pub_state_on_terminate)

    async def checkpoint(self) -> InternalObservation:
        snapshot = await self.engine._serialize_engine()
        return {
            "engine_snapshot": snapshot.model_dump(),
            "message": "Checkpoint created",
        }

    async def _obs(self, priv: Dict[str, Any], pub: Dict[str, Any]):
        if self.custom_obs:
            return await self.custom_obs.get_observation(pub, priv)
        return {**pub, **priv}
