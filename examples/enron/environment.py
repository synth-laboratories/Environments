# environment.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from src.environment.tools import EnvToolCall
from src.environment.shared_engine import GetObservationCallable, InternalObservation
from src.stateful.core import StatefulEnvironment
from examples.enron.engine import (
    EnronEngine,
    ACTION_SEARCH,
    ACTION_READ,
    ACTION_ANSWER,
)
from examples.enron.taskset import EnronTaskInstance


# -------- pydantic schemas (used by agent / LLM function calls)
class SearchEmailsArgs(BaseModel):
    inbox: str = Field(..., description="Email address performing the search")
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


# --------------------------------------------------------------------------- tool wrappers
class SearchEmails(EnvToolCall):
    def __init__(self, **kwargs):
        self.action = (ACTION_SEARCH, kwargs)


class ReadEmail(EnvToolCall):
    def __init__(self, message_id: str):
        self.action = (ACTION_READ, message_id)


class AnswerQuestion(EnvToolCall):
    def __init__(self, answer: str):
        self.action = (ACTION_ANSWER, answer)


# -- terminate wrapper (maps to an empty-answer ACTION_ANSWER) --------------
class Terminate(EnvToolCall):
    def __init__(self):
        self.action = (ACTION_ANSWER, "")


# -------- observation callable (optional for formatted observations)
class SynthEnronObservationCallable(GetObservationCallable):
    async def get_observation(self, pub: Dict[str, Any], priv: Dict[str, Any]) -> InternalObservation:
        """Format observation as a human-readable string."""
        q = pub.get("question")
        rwd = priv.get("reward_last")
        return f"Q: {q}\nReward Δ: {rwd}"


# --------------------------------------------------------------------------- environment
class EnronEnvironment(StatefulEnvironment):
    def __init__(self, task_instance: EnronTaskInstance, custom_obs: Optional[GetObservationCallable] = None):
        self.engine = EnronEngine(task_instance)
        self.custom_obs = custom_obs
        self.name = "Enron-QA-Env"

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._obs(priv, pub)

    async def step(
        self,
        calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]],
    ) -> InternalObservation:
        # normalise → always [[EnvToolCall]]
        if isinstance(calls, EnvToolCall):
            calls = [[calls]]
        elif calls and isinstance(calls[0], EnvToolCall):
            calls = [calls]

        priv, pub = await self.engine._step_engine(calls[0][0].action)
        return await self._obs(priv, pub)

    async def checkpoint(self) -> InternalObservation:
        return await self.engine._serialize_engine()

    async def _obs(self, priv: Dict[str, Any], pub: Dict[str, Any]):
        if self.custom_obs:
            return await self.custom_obs.get_observation(pub, priv)
        return {**pub, **priv}
